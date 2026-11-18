"""Encoder/decoder round-trips for the ZMQ control messages (no GPU).

Every message that crosses api -> tokenizer -> scheduler -> tokenizer -> api must survive the
wire with its fields intact; these pin the ones carrying state a later consumer reads back
(rebuild control, prompt admission, per-reply token deltas and KV usage).
"""

from __future__ import annotations

import pytest
import torch

from freetoken.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    MMItem,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    CacheRebuildBackendMsg,
    CacheRebuildMsg,
    CacheRebuildReply,
    CacheRebuildResultMsg,
    PromptAdmittedMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
from freetoken.core import SamplingParams


def test_executor_batch_wire_matches_independent_header_and_keeps_uid_zero():
    import socket
    import struct
    from freetoken.engine.backend import (
        ExecutionWindow, Phase, RequestKey, ResourceHandle, StateCursor,
    )
    from freetoken.native.executor_protocol import Kind, decode_step, encode_step, recv_frame

    state, executable = ResourceHandle(7, 2, 1, 1), ResourceHandle(7, 2, 2, 1)
    window = ExecutionWindow(RequestKey(0, 3), 7, state, StateCursor(0, 0))
    plan = window.grant(executable, Phase.PREFILL, 1, 4, input_ids=(0, 248319, 42))
    payload = encode_step(plan)
    left, right = socket.socketpair()
    try:
        # Literal layout is independent of the protocol encoder.
        left.sendall(struct.pack("<IHHIIQQ", 0x32585446, 3, 4, len(payload), 0, 0, 3) + payload)
        kind, uid, epoch, encoded = recv_frame(right)
        assert (kind, uid, epoch) == (Kind.SUBMIT, 0, 3)
        assert decode_step(encoded) == plan
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("cancel", [False, True])
def test_executor_credit_drain_keeps_out_of_order_gpu_state_alive(cancel):
    from dataclasses import replace
    from freetoken.engine.backend import (
        Completion, ExecutionWindow, Phase, RequestKey, ResourceHandle, StateCursor,
    )

    state, executable = ResourceHandle(1, 1, 1, 1), ResourceHandle(1, 1, 2, 1)
    initial = StateCursor(5, 4, 42)
    window = ExecutionWindow(RequestKey(0, 1), 1, state, initial, limit=2)
    first = window.grant(executable, Phase.MTP, 4, 11, draft_depth=3)
    second = window.grant(executable, Phase.MTP, 4, 17, draft_depth=3)
    middle, final = StateCursor(7, 6, 19), StateCursor(8, 7, 20)
    c1 = Completion(first.ticket, replace(state, version=2), initial, middle,
                    (18, 19), (True, False))
    c2 = Completion(second.ticket, replace(state, version=3), middle, final, (20,), (False,))
    if cancel:
        window.revoke()
    assert window.complete(c2) == ()
    assert not window.drained
    assert window.cursor == initial
    assert window.complete(c1) == (() if cancel else (c1, c2))
    assert window.drained
    assert window.cursor == (initial if cancel else final)
    with pytest.raises(ValueError):
        window.complete(c1)


def test_executor_terminal_trimming_retires_physical_state_before_reuse():
    from dataclasses import replace
    from freetoken.engine.backend import (
        Completion, ExecutionWindow, Phase, RequestKey, ResourceHandle, StateCursor,
    )
    from freetoken.native.executor_protocol import decode_completion, encode_completion

    state, executable = ResourceHandle(1, 1, 1, 1), ResourceHandle(1, 1, 2, 1)
    cursor = StateCursor(10, 9, 42)
    window = ExecutionWindow(RequestKey(0, 1), 1, state, cursor)
    with pytest.raises(ValueError):
        window.grant(executable, Phase.MTP, 1, 11, draft_depth=3)
    with pytest.raises(ValueError):
        window.grant(executable, Phase.MTP, 1, 14, draft_depth=3)
    plan = window.grant(executable, Phase.MTP, 1, 16, draft_depth=3)
    result = Completion(plan.ticket, replace(state, version=2), cursor, None, (43,), (True,),
                        terminal_frontier=11, terminal_reason="length",
                        retired_computed_tokens=13, proposed_drafts=3,
                        verified_drafts=3, accepted_drafts=2)
    with pytest.raises(ValueError, match="accepted drafts"):
        replace(result, accepted_drafts=3, verified_drafts=2)
    assert decode_completion(encode_completion(result)) == result
    assert window.complete(result) == (result,)
    assert window.terminal_frontier == 11 and window.revoked and window.drained
    with pytest.raises(RuntimeError):
        window.grant(executable, Phase.DECODE, 1, 12)


def test_executor_reservation_rebases_after_actual_speculative_acceptance():
    from dataclasses import replace
    from freetoken.engine.backend import (
        Completion, ExecutionWindow, Phase, RequestKey, ResourceHandle, StateCursor,
    )

    state, executable = ResourceHandle(1, 1, 1, 1), ResourceHandle(1, 1, 2, 1)
    initial = StateCursor(5, 4, 42)
    window = ExecutionWindow(RequestKey(0, 1), 1, state, initial, limit=2)
    first = window.grant(executable, Phase.MTP, 4, 11, draft_depth=3)
    window.grant(executable, Phase.MTP, 4, 17, draft_depth=3)
    actual = StateCursor(6, 5, 43)
    window.complete(Completion(first.ticket, replace(state, version=2),
                              initial, actual, (43,), (False,)))
    # The unaccepted three-token estimate must not accumulate in every future reservation.
    assert window.reservation_floor == 12
    next_plan = window.grant(executable, Phase.MTP, 4, 18, draft_depth=3)
    assert next_plan.reserved_through == 18


@pytest.mark.parametrize("payload", [
    b'{"step_id":1,"step_id":2}', b'{"scale":NaN}', b'{"scale":Infinity}',
])
def test_executor_refuses_ambiguous_or_nonfinite_wire_metadata(payload):
    from freetoken.native.executor_protocol import decode

    with pytest.raises(ValueError):
        decode(payload)


@pytest.mark.parametrize("committed,computed,pending", [(5, 3, 42), (5, 5, 42), (5, 4, None)])
def test_executor_distinguishes_sampled_and_computed_state_frontiers(committed, computed, pending):
    from freetoken.engine.backend import StateCursor

    with pytest.raises(ValueError):
        StateCursor(committed, computed, pending)


@pytest.mark.parametrize("count", [0, 1, 4])
@pytest.mark.parametrize("terminal", [False, True])
def test_native_committed_tokens_keep_epoch_and_binary_ids(count, terminal):
    import socket
    import struct
    from freetoken.native.protocol import Kind, decode_tokens, recv_frame, send_frame

    left, right = socket.socketpair()
    try:
        tokens = [0, 248044, 248319, 42][:count]
        payload = struct.pack("<QI", 123456789, count) + struct.pack(f"<{count}i", *tokens)
        kind = Kind.FINAL_TOKENS if terminal else Kind.TOKENS
        send_frame(left, kind, 77, 3, payload)
        kind, uid, epoch, actual = recv_frame(right)
        assert (kind, uid, epoch) == (Kind.FINAL_TOKENS if terminal else Kind.TOKENS, 77, 3)
        assert decode_tokens(actual) == (123456789, tokens)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("field,value", [(0, 0), (1, 65535), (2, 65535), (3, 8 * 1024 * 1024 + 1), (4, 1)])
def test_native_rejects_incompatible_or_unbounded_header(field, value):
    import socket
    from freetoken.native.protocol import HEADER, MAGIC, VERSION, Kind, recv_frame

    fields = [MAGIC, VERSION, Kind.TOKENS, 0, 0, 1, 1]
    fields[field] = value
    left, right = socket.socketpair()
    try:
        left.sendall(HEADER.pack(*fields))
        with pytest.raises(ValueError):
            recv_frame(right)
    finally:
        left.close()
        right.close()


def test_native_partial_header_eof_fails_instead_of_waiting_for_worker():
    import socket
    from freetoken.native.protocol import recv_frame

    left, right = socket.socketpair()
    try:
        left.sendall(b"FTNW\x01")
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            recv_frame(right)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("payload", [b"", b"0" * 11, b"\0" * 12 + b"1234"])
def test_native_rejects_malformed_committed_token_count(payload):
    from freetoken.native.protocol import decode_tokens

    with pytest.raises(ValueError):
        decode_tokens(payload)


def test_inline_system_policy_tokenizer_roundtrip():
    for policy in (None, "auto", "preserve", "fold"):
        msg = TokenizeMsg(uid=1, text=[{"role": "system", "content": "instruction"}],
                          sampling_params=SamplingParams(), inline_system_policy=policy)
        decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert decoded.inline_system_policy == policy
        assert decoded.text == msg.text
    legacy = BaseTokenizerMsg.encoder(msg)
    legacy.pop("inline_system_policy")
    assert BaseTokenizerMsg.decoder(legacy).inline_system_policy is None


def test_cache_rebuild_msg_roundtrip():
    msg = CacheRebuildMsg(request_id="abc", moe_cache_size=8, num_pages=1024, mode="if_idle")
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("abc", 8, 1024, "if_idle")


def test_cache_rebuild_backend_msg_roundtrip():
    msg = CacheRebuildBackendMsg(request_id="r1", moe_cache_size=None, num_pages=256, mode="drain")
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, CacheRebuildBackendMsg)
    assert (out.request_id, out.moe_cache_size, out.num_pages, out.mode) == ("r1", None, 256, "drain")


def test_cache_rebuild_result_msg_roundtrip():
    msg = CacheRebuildResultMsg(request_id="r2", status="ok", moe_cache_size=16, num_pages=512)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, CacheRebuildResultMsg)
    assert (out.request_id, out.status, out.moe_cache_size, out.num_pages, out.error) == (
        "r2", "ok", 16, 512, None,
    )


def test_cache_rebuild_reply_roundtrip():
    msg = CacheRebuildReply(request_id="r3", status="failed", error="boom")
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))
    assert isinstance(out, CacheRebuildReply)
    assert (out.request_id, out.status, out.error) == ("r3", "failed", "boom")


def test_prompt_admitted_msg_roundtrip():
    msg = PromptAdmittedMsg(uid=42, prompt_tokens=1234, cached_tokens=500)
    out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(out, PromptAdmittedMsg)
    assert (out.uid, out.prompt_tokens, out.cached_tokens) == (42, 1234, 500)


def test_user_reply_token_deltas_round_trip():
    msg = UserReply(
        uid=7,
        incremental_output="hello",
        finished=False,
        prompt_tokens_delta=11,
        completion_tokens_delta=3,
        cached_tokens=4,
        kv_used_pages=40,
        kv_total_pages=512,
        gpu_mem_bytes=64 * (1 << 30),
        native_stats={"memory_kind": "native owned", "ranks": [{"native_committed_bytes": 12345}]},
    )

    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(msg))

    assert isinstance(decoded, UserReply)
    assert decoded.uid == 7
    assert decoded.incremental_output == "hello"
    assert decoded.finished is False
    assert decoded.prompt_tokens_delta == 11
    assert decoded.completion_tokens_delta == 3
    assert decoded.cached_tokens == 4
    assert decoded.kv_used_pages == 40
    assert decoded.kv_total_pages == 512
    assert decoded.gpu_mem_bytes == 64 * (1 << 30)
    assert decoded.native_stats == msg.native_stats


def test_detokenize_msg_carries_kv_usage_round_trip():
    msg = DetokenizeMsg(
        uid=3, next_tokens=[42], finished=True,
        kv_used_pages=10, kv_total_pages=256, gpu_mem_bytes=1 << 30,
        mamba_used_slots=7, mamba_total_slots=64,
        swa_used_tokens=8448, swa_total_tokens=76800,
        native_stats={"memory_kind": "native owned", "ranks": [{"native_committed_bytes": 12345}]},
    )
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert (decoded.kv_used_pages, decoded.kv_total_pages, decoded.gpu_mem_bytes) == (10, 256, 1 << 30)
    assert (decoded.mamba_used_slots, decoded.mamba_total_slots) == (7, 64)
    assert (decoded.swa_used_tokens, decoded.swa_total_tokens) == (8448, 76800)
    assert decoded.native_stats == msg.native_stats


def test_terminal_replies_carry_scheduler_timings_over_the_wire():
    timings = {"queue_ms": 5.5, "cache_n": 8, "prompt_n": 2, "prompt_ms": 100.0,
               "predicted_n": 4, "predicted_ms": 250.0, "draft_n": 6, "draft_n_accepted": 3}
    msg = DetokenizeMsg(uid=3, next_tokens=[42], finished=True, timings=timings)
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert isinstance(decoded, DetokenizeMsg)
    assert decoded.timings == timings

    reply = UserReply(uid=3, incremental_output="", finished=True,
                      completion_tokens_delta=1, timings=timings)
    out = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert isinstance(out, UserReply)
    assert out.timings == timings
    # intermediate replies and aborts/errors never carry timings
    plain = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(
        UserReply(uid=3, incremental_output="x", finished=False)
    ))
    assert plain.timings is None


def test_client_dicts_with_the_wire_tag_key_survive_intact():
    """Tool JSON Schemas and chat_template_kwargs are free-form client data. A field literally
    named ``__type__`` (a common discriminator) must not be read back as a serialized class --
    that used to kill the tokenizer worker on an unknown/incompatible name."""
    hostile = [
        {"__type__": "AbortMsg"},                                    # a real class name
        {"__type__": "NoSuchClassAnywhere"},                         # an unknown one
        {"type": "object", "properties": {"__type__": {"type": "string"}}},
        {"__raw_dict__": {"a": 1}},                                  # collides with the escape key
        {"deep": {"__type__": "AbortMsg", "l": [{"__type__": "x"}]}},
    ]
    for payload in hostile:
        msg = TokenizeMsg(
            uid=1, text="hi", sampling_params=SamplingParams(),
            chat_template_kwargs=payload,
            tools=[{"type": "function", "function": {"name": "f", "parameters": payload}}],
        )
        out = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
        assert isinstance(out, TokenizeMsg)
        assert out.chat_template_kwargs == payload
        assert out.tools[0]["function"]["parameters"] == payload


def test_tensor_wire_nd_and_dtype_roundtrip():
    import numpy as np
    import torch
    from freetoken.message.utils import deserialize_type, serialize_type

    cases = [
        torch.arange(6, dtype=torch.int32),
        torch.arange(12, dtype=torch.int64).reshape(3, 4),
        torch.randn(2, 3, 5, dtype=torch.float32),
        torch.randn(4, 1536, dtype=torch.bfloat16),
        torch.randn(3, 4)[:, :2],  # non-contiguous input
        torch.tensor(7, dtype=torch.int32),  # 0-d
    ]
    for t in cases:
        out = deserialize_type({}, serialize_type(t))
        assert out.dtype == t.dtype
        assert out.shape == t.shape
        assert torch.equal(out, t)

    # legacy payloads carry no shape and decode as 1-D
    legacy = {
        "__type__": "Tensor",
        "buffer": np.arange(4, dtype=np.int32).tobytes(),
        "dtype": "torch.int32",
    }
    out = deserialize_type({}, legacy)
    assert out.shape == (4,) and out.dtype == torch.int32

    # 1-D payloads still omit the shape field (old decoders can read them)
    assert "shape" not in serialize_type(torch.arange(3, dtype=torch.int32))


def test_user_msg_with_mm_items_survives_the_wire():
    item = MMItem(
        modality="image",
        hash=0x1234ABCD,
        pad_value=1_000_000 + 0x1234ABCD,
        offsets=[[7, 11]],
        feature=torch.randn(16, 1536, dtype=torch.bfloat16),
        model_specific_data={"grid_thw": [1, 4, 4]},
    )
    msg = UserMsg(
        uid=9,
        input_ids=torch.arange(16, dtype=torch.int32),
        sampling_params=SamplingParams(),
        mm_items=[item],
        mrope_positions=torch.zeros(3, 16, dtype=torch.int32),
        mrope_delta=-3,
    )
    out = BaseBackendMsg.decoder(msg.encoder())
    assert isinstance(out, UserMsg)
    got = out.mm_items[0]
    assert (got.hash, got.pad_value, got.offsets) == (0x1234ABCD, 1_000_000 + 0x1234ABCD, [[7, 11]])
    assert got.num_tokens == 4 and got.grid_thw == [1, 4, 4]  # model_specific_data reads as attributes
    assert got.precomputed_embeddings is None
    assert torch.equal(got.feature, item.feature)
    assert out.mrope_positions.shape == (3, 16) and out.mrope_delta == -3


def test_mm_item_shape_rules():
    t = torch.zeros(2, 4)
    item = MMItem(modality="image", hash=1, pad_value=1, offsets=[[3, 5], [9, 12]], feature=t)
    item.validate()
    assert item.num_tokens == 5
    assert getattr(item, "grid_thw", None) is None  # missing model-specific key -> AttributeError path
    with pytest.raises(ValueError, match="half-open"):
        MMItem(modality="image", hash=1, pad_value=1, offsets=[[5, 5]], feature=t).validate()
    with pytest.raises(ValueError, match="exactly one"):
        MMItem(modality="image", hash=1, pad_value=1, offsets=[[0, 2]]).validate()
    with pytest.raises(ValueError, match="exactly one"):
        MMItem(
            modality="image", hash=1, pad_value=1, offsets=[[0, 2]], feature=t, precomputed_embeddings=t
        ).validate()
