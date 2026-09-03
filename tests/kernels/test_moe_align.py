"""Tests for freetoken.kernel.triton.moe_align.moe_align_block_size.

This is the AMD / no-sgl_kernel fallback that decides which expert handles which
block of tokens, and until now it had NO direct test coverage. Triton 3.8.0's AMD
backend miscompiled its kernels on gfx1101, producing structurally wrong routing
(up to 92% of downstream output elements wrong) and GPU memory faults -- every
existing test only exercised this module indirectly, so nothing pointed at the
real culprit. A naive pure-torch replacement then broke HIP graph capture, because
it sized buffers from tensor *contents* instead of shape alone. This file locks
down both properties: correctness (against a pure-Python reference), and
capture-safety (a real torch.cuda.graph capture + replay with mutated inputs).

The correctness tests target _moe_align_torch directly: both Triton paths leave
expert_ids past the used region uninitialized (torch.empty + masked stores), and
the small-batch kernel also sorted_token_ids past num_tokens_post_pad, so the
strict fill assertions here apply only to the torch implementation.
"""

from __future__ import annotations

import pytest
import torch

import freetoken.kernel.triton.moe_align as moe_align_mod
from freetoken.kernel.triton.moe_align import _moe_align_torch, moe_align_block_size

# _moe_align_torch is pure torch: identical code path on CPU, CUDA, and ROCm.
# Running the correctness suite on CPU keeps it green on every CI runner;
# the cuda parametrization adds real-GPU coverage when a GPU is present.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return request.param


def _buffer_sizes(numel: int, block_size: int, num_experts: int) -> tuple[int, int]:
    """Mirrors the buffer-size formula in moe_align.py's module docstring exactly."""
    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_m_blocks = -(-max_num_tokens_padded // block_size)  # ceil div
    return max_num_tokens_padded, max_num_m_blocks


def _tokens_by_expert(topk_ids: torch.Tensor, num_experts: int) -> list[list[int]]:
    """buckets[e] = flat token indices routed to expert e; invalid entries dropped."""
    effective_E = num_experts + 1
    flat = topk_ids.reshape(-1).tolist()
    buckets: list[list[int]] = [[] for _ in range(effective_E)]
    for tok_idx, e in enumerate(flat):
        if 0 <= e < effective_E:
            buckets[e].append(tok_idx)
    return buckets


def _expert_layout(topk_ids: torch.Tensor, block_size: int, num_experts: int):
    """Per-expert token buckets, counts, padded-block counts, and start offsets
    (in token units). Shared by the reference builder and the output checker below
    so both use the exact same layout math."""
    buckets = _tokens_by_expert(topk_ids, num_experts)
    counts = [len(b) for b in buckets]
    nblk = [-(-c // block_size) for c in counts]  # ceil div
    starts: list[int] = []
    blocks_so_far = 0
    for n in nblk:
        starts.append(blocks_so_far * block_size)
        blocks_so_far += n
    return buckets, counts, nblk, starts


def _reference(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[list[int], list[int], int]:
    """Pure-Python ground truth for moe_align_block_size (see moe_align.py's module
    docstring for the authoritative semantics). Deliberately slow and obvious rather
    than clever: nested loops over token buckets, not vectorized."""
    numel = topk_ids.numel()
    effective_E = num_experts + 1
    buckets, counts, nblk, starts = _expert_layout(topk_ids, block_size, num_experts)
    num_tokens_post_pad = sum(nblk) * block_size
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)

    sorted_token_ids = [numel] * max_num_tokens_padded  # sentinel-filled
    expert_ids = [0] * max_num_m_blocks
    for e in range(effective_E):
        s = starts[e]
        for i, tok in enumerate(buckets[e]):
            sorted_token_ids[s + i] = tok
        for b in range(nblk[e]):
            expert_ids[s // block_size + b] = e

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def _assert_outputs_match_reference(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Exact match on num_tokens_post_pad and expert_ids; per-expert token SETS
    (order within an expert is explicitly unspecified -- atomicAdd-ranked scatter);
    every slot not claimed by any expert holds the sentinel (== numel)."""
    numel = topk_ids.numel()
    effective_E = num_experts + 1
    _, ref_expert_ids, ref_npp = _reference(topk_ids, block_size, num_experts)

    assert int(num_tokens_post_pad.item()) == ref_npp
    assert expert_ids.tolist() == ref_expert_ids

    _, counts, _, starts = _expert_layout(topk_ids, block_size, num_experts)
    buckets = _tokens_by_expert(topk_ids, num_experts)
    got = sorted_token_ids.tolist()
    claimed: set[int] = set()
    for e in range(effective_E):
        s, c = starts[e], counts[e]
        assert sorted(got[s : s + c]) == sorted(buckets[e]), f"expert {e} token set mismatch"
        claimed.update(range(s, s + c))

    sentinel = numel
    for i, v in enumerate(got):
        if i not in claimed:
            assert v == sentinel, f"slot {i} should hold sentinel {sentinel}, got {v}"


def _assert_matches_reference(topk_ids: torch.Tensor, block_size: int, num_experts: int) -> None:
    numel = topk_ids.numel()
    sorted_token_ids, expert_ids, num_tokens_post_pad = _moe_align_torch(
        topk_ids, block_size, num_experts
    )
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)

    assert sorted_token_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.dtype == torch.int32
    assert sorted_token_ids.shape == (max_num_tokens_padded,)
    assert expert_ids.shape == (max_num_m_blocks,)
    assert num_tokens_post_pad.shape == (1,)

    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_token_ids, expert_ids, num_tokens_post_pad
    )


@pytest.mark.parametrize(
    "numel,num_experts,block_size",
    [
        (1, 8, 16),
        (7, 8, 16),
        (64, 8, 16),
        (256, 64, 32),
        (1536, 256, 64),
        (300, 4, 32),
    ],
)
def test_matches_reference(numel, num_experts, block_size, device):
    torch.manual_seed(0)
    effective_E = num_experts + 1
    topk_ids = torch.randint(0, effective_E, (numel,), device=device, dtype=torch.int32)
    _assert_matches_reference(topk_ids, block_size, num_experts)


def test_empty_input(device):
    torch.manual_seed(0)
    num_experts, block_size = 8, 16
    topk_ids = torch.empty(0, dtype=torch.int32, device=device)

    sorted_token_ids, expert_ids, num_tokens_post_pad = _moe_align_torch(
        topk_ids, block_size, num_experts
    )
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(0, block_size, num_experts)

    assert sorted_token_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.dtype == torch.int32
    assert sorted_token_ids.shape == (max_num_tokens_padded,)
    assert expert_ids.shape == (max_num_m_blocks,)
    assert num_tokens_post_pad.shape == (1,)
    assert int(num_tokens_post_pad.item()) == 0
    # buffers are 0-length for an empty input (numel*block_size == 0), so "fully
    # sentinel/zero filled" is vacuously true -- the shapes above are the real check.


def test_invalid_expert_ids_are_ignored(device):
    num_experts, block_size = 4, 16
    effective_E = num_experts + 1
    # mix of valid ids [0, num_experts] with invalid ones: negative and >= effective_E
    values = [0, 1, 2, 3, 4, -1, -2, 5, 100, 3, 0, -1, 2, 4, 5, 1000, 2, 3, 0, 1]
    topk_ids = torch.tensor(values, dtype=torch.int32, device=device)

    # full reference comparison: this also proves invalid entries don't shift where
    # the valid tokens land, since the reference is computed on this same input and
    # already drops invalid entries the same way the kernel must.
    _assert_matches_reference(topk_ids, block_size, num_experts)

    sorted_token_ids, _, _ = _moe_align_torch(topk_ids, block_size, num_experts)
    invalid_tok_idx = {i for i, e in enumerate(values) if not (0 <= e < effective_E)}
    present = set(sorted_token_ids.tolist())
    assert not (invalid_tok_idx & present), "an invalid entry's token index leaked into output"


def test_zero_experts(device):
    """num_experts == 0 still leaves effective_E == 1 (the sentinel slot), so an
    input of all-zero ids routes everything to that one slot."""
    num_experts, block_size = 0, 16
    topk_ids = torch.zeros(8, dtype=torch.int32, device=device)
    _assert_matches_reference(topk_ids, block_size, num_experts)


def test_all_invalid_ids(device):
    """Nonempty input where every id is invalid: nothing is routed, so the padded
    count is 0, expert_ids is all zeros, and every sorted slot holds the sentinel
    (all invalid tokens collide on the single trash slot past the real buffer)."""
    num_experts, block_size = 4, 16
    values = [-3, -1, 5, 6, 99, 1000, -7, 5]  # all outside [0, effective_E)
    topk_ids = torch.tensor(values, dtype=torch.int32, device=device)

    _assert_matches_reference(topk_ids, block_size, num_experts)

    sorted_token_ids, expert_ids, num_tokens_post_pad = _moe_align_torch(
        topk_ids, block_size, num_experts
    )
    assert int(num_tokens_post_pad.item()) == 0
    assert (expert_ids == 0).all()
    assert (sorted_token_ids == topk_ids.numel()).all()  # sentinel everywhere


def test_expert_ids_past_used_region_are_zero(device):
    """fused_moe_kernel loads expert_ids[pid_m] and multiplies it into a base
    pointer to address that expert's weight slice. A junk (nonzero, out-of-range)
    value past the used region therefore produces a wild address and a GPU memory
    fault -- every slot beyond num_tokens_post_pad//block_size must be exactly 0."""
    torch.manual_seed(5)
    num_experts, block_size, numel = 64, 32, 256
    topk_ids = torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)

    _, expert_ids, num_tokens_post_pad = _moe_align_torch(topk_ids, block_size, num_experts)
    used_blocks = int(num_tokens_post_pad.item()) // block_size
    assert (expert_ids[used_blocks:] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph capture needs a GPU")
def test_is_graph_capturable():
    """Regression guard for the capture bug this file exists to catch. Any data-
    dependent shape (boolean-mask indexing, bincount, nonzero, .item(),
    repeat_interleave-by-tensor) forces a device->host sync, and torch.cuda.graph
    raises "operation not permitted when stream is capturing" the moment that
    happens during capture. moe_align_block_size's buffer shapes depend only on
    numel/block_size/num_experts -- never on topk_ids' contents -- so capture must
    succeed, and replaying with genuinely different routing must still produce
    correct output.

    Warmup/capture pattern copied from
    tests/moe/test_fused_moe.py:190-209
    (test_fused_experts_grouped_impl_is_cuda_graph_capturable): run the op into a
    stable pre-allocated output via .copy_() a few times on the default stream,
    synchronize, then capture that same run() under torch.cuda.graph.
    """
    device = torch.device("cuda")
    num_experts, block_size, numel = 8, 16, 32
    torch.manual_seed(11)

    topk_ids = torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)
    max_num_tokens_padded, max_num_m_blocks = _buffer_sizes(numel, block_size, num_experts)
    sorted_out = torch.empty(max_num_tokens_padded, dtype=torch.int32, device=device)
    expert_out = torch.empty(max_num_m_blocks, dtype=torch.int32, device=device)
    npp_out = torch.empty(1, dtype=torch.int32, device=device)

    def run():
        st, ei, npp = _moe_align_torch(topk_ids, block_size, num_experts)
        sorted_out.copy_(st)
        expert_out.copy_(ei)
        npp_out.copy_(npp)

    for _ in range(3):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_out, expert_out, npp_out
    )

    # replay with genuinely different routing -- only the captured input tensor's
    # CONTENTS change (via copy_), never its shape, since that's what capture allows.
    torch.manual_seed(23)
    topk_ids.copy_(
        torch.randint(0, num_experts + 1, (numel,), device=device, dtype=torch.int32)
    )
    graph.replay()
    torch.cuda.synchronize()
    _assert_outputs_match_reference(
        topk_ids, block_size, num_experts, sorted_out, expert_out, npp_out
    )


def test_dispatch_routes_by_platform(monkeypatch):
    """moe_align_block_size must route to the torch impl on ROCm and the Triton
    impl everywhere else. Implementations are stubbed so this runs anywhere.
    Patching the module-local is_rocm symbol deliberately bypasses the real
    functools.cache'd detector: this tests dispatcher branching, not platform
    detection (which tests/utils/test_rocm_arch.py covers)."""
    calls: list[str] = []
    fake = (object(), object(), object())
    monkeypatch.setattr(
        moe_align_mod, "_moe_align_torch", lambda *a: (calls.append("torch"), fake)[1]
    )
    monkeypatch.setattr(
        moe_align_mod, "_moe_align_triton", lambda *a: (calls.append("triton"), fake)[1]
    )
    topk_ids = torch.zeros(4, dtype=torch.int32)

    monkeypatch.setattr(moe_align_mod, "is_rocm", lambda: True)
    assert moe_align_block_size(topk_ids, 16, 8) is fake
    monkeypatch.setattr(moe_align_mod, "is_rocm", lambda: False)
    assert moe_align_block_size(topk_ids, 16, 8) is fake
    assert calls == ["torch", "triton"]
