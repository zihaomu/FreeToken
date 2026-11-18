"""Scheduling and token-commit policy shared by execution backends.

Device streams, tensor storage, and batch submission belong to the adapters.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    import torch
    from freetoken.core import Req
    from freetoken.message import DetokenizeMsg


def prefill_chunk_size(cached_len: int, remaining: int, budget: int, alignment: int = 1) -> int:
    size = min(budget, remaining)
    if alignment > 1 and 0 < size < remaining:
        aligned = (cached_len + size) // alignment * alignment - cached_len
        size = aligned if aligned > 0 else size
    return size


def additional_page_tokens(total_len: int, cached_len: int, page_size: int) -> int:
    return ((total_len + page_size - 1) // page_size -
            (cached_len + page_size - 1) // page_size) * page_size


class SchedulerCore:
    @staticmethod
    def select_batch(prefill_manager, decode_manager, prefill_budget):
        return (prefill_manager.schedule_next_batch(prefill_budget)
                or decode_manager.schedule_next_batch())

    def _commit_tokens(self, req: Req, tokens: torch.Tensor) -> Tuple[DetokenizeMsg, int]:
        """Append sampled tokens in order until one finishes the request; returns the reply and how many tokens were committed."""
        from freetoken.message import DetokenizeMsg

        params = req.sampling_params
        committed: List[int] = []
        finished, finish_reason, matched_stop = False, None, None
        for j in range(tokens.numel()):
            req.append_host(tokens[j : j + 1])
            token = int(tokens[j].item())
            committed.append(token)
            # EOS / stop-string -> "stop", output budget exhausted -> "length";
            # EOS and stop strings win over length.
            # Overlap can advance device_len one token ahead of this host reply.
            hit_length = req.input_ids.numel() >= req.max_device_len
            hit_eos = not params.ignore_eos and token in self.eos_token_ids
            matched_stop = self._match_stop_str(req) if not hit_eos and params.stop_strs else None
            if hit_length or hit_eos or matched_stop is not None:
                finished = True
                finish_reason = "stop" if (hit_eos or matched_stop is not None) else "length"
                break
            if token == self.toolcall_anchor_id and req.toolcall_anchor_len is None:
                req.toolcall_anchor_len = req.input_ids.numel()
        msg = DetokenizeMsg(
            uid=req.uid,
            next_tokens=committed,
            finished=finished,
            finish_reason=finish_reason,
            matched_stop=matched_stop,
            stop_strs=params.stop_strs or None,
        )
        return msg, len(committed)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None
