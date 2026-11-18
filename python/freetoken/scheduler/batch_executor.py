"""CPU scheduler adapter for explicitly authorized device batches.

This shares request/token semantics with the Torch scheduler. A worker receives
bounded steps; it never owns the request queue or chooses the next prefill chunk.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from freetoken.core import Batch, Req
from freetoken.engine.backend import Completion, ExecutionWindow, Phase, RequestKey
from freetoken.kvcache.storage import BatchStateStorage, StateLease
from freetoken.message import DetokenizeMsg, ErrorReplyMsg, PromptAdmittedMsg
from .core import SchedulerCore, prefill_chunk_size
from .decode import DecodeManager


@dataclass
class _Active:
    request: Req
    key: RequestKey
    window: ExecutionWindow
    prompt: tuple[int, ...]
    lease: StateLease
    offset: int = 0
    terminal_recorded: bool = False
    terminal_message: object | None = None
    counters: dict = field(default_factory=lambda:dict(proposed_drafts=0,verified_drafts=0,
        sampler_accepted_drafts=0,committed_drafts=0,committed_tokens=0,
        authorized_steps=0,retired_credits=0,committed_steps=0))


class _PrefillSource:
    def __init__(self, adapter):
        self.adapter = adapter

    def schedule_next_batch(self, budget):
        active = self.adapter.active
        if active is not None and active.offset < len(active.prompt):
            return Batch(reqs=[active.request], phase="prefill")
        return None


class BatchSchedulerAdapter(SchedulerCore):
    def __init__(self, client, tokenizer, eos_token_ids, *, credit_window=2):
        if client.ready.get("scheduler_owner") != "freetoken":
            raise ValueError("explicit FreeToken batch worker required")
        if credit_window not in (1, 2):
            raise ValueError("one or two bounded credits required")
        self.client, self.tokenizer = client, tokenizer
        self.eos_token_ids = frozenset(eos_token_ids)
        self.toolcall_anchor_id = None
        self.credit_window = credit_window
        self.executable = client.compile_composition()
        self.storage = BatchStateStorage(client, credit_window=credit_window)
        self.decode_manager = DecodeManager(page_size=4, token_headroom=client.config.mtp_depth)
        self.active: _Active | None = None
        self.prefill_manager = _PrefillSource(self)
        self._epoch = 0

    def admit(self, message):
        enqueued = time.perf_counter()
        if self.active is not None:
            return [ErrorReplyMsg(message.uid, "batch executor is busy (batch 1)")]
        if message.mm_items or message.mrope_positions is not None:
            return [ErrorReplyMsg(message.uid, "batch executor supports text requests only")]
        prompt = tuple(int(t) for t in message.input_ids.tolist())
        room = self.client.config.max_context - len(prompt)
        if not prompt or room < 1 or message.sampling_params.max_tokens < 1:
            return [ErrorReplyMsg(message.uid, "prompt/output exceeds the context", "context_length_exceeded")]
        params = message.sampling_params
        limit = min(params.max_tokens, room)
        self._epoch += 1
        key = RequestKey(message.uid, self._epoch)
        lease = self.storage.reserve(key, prompt, max_tokens=limit, sampling_params=params)
        cached = lease.window.cursor.computed_tokens
        req = Req(message.input_ids, 0, cached, limit, message.uid, params, cache_handle=lease.handle)
        now = time.perf_counter()
        req.stats.enqueued_at, req.stats.first_chunk_at = enqueued, now
        req.stats.prompt_n, req.stats.cache_n = len(prompt)-cached, cached
        self.active = _Active(req, key, lease.window, prompt, lease, offset=cached)
        self.decode_manager.filter_reqs([req])
        self.refill()
        return [PromptAdmittedMsg(message.uid, len(prompt), cached)]

    def refill(self):
        active = self.active
        if active is None or active.window.revoked:
            return
        req, window = active.request, active.window
        while len(window.pending) < self.credit_window:
            batch = self.select_batch(self.prefill_manager, self.decode_manager,
                                      self.client.config.prefill_chunk)
            if batch is None:
                break
            pending = window.plans
            remaining = req.max_device_len - req.input_ids.numel() - sum(p.max_committed for p in pending)
            if batch.is_prefill:
                size = prefill_chunk_size(active.offset, len(active.prompt) - active.offset,
                                          self.client.config.prefill_chunk, 4)
                chunk = active.prompt[active.offset:active.offset + size]
                output = int(active.offset + size == len(active.prompt))
                if output and remaining < 1:
                    break
                phase = Phase.PREFILL
            else:
                if remaining < 1:
                    break
                phase = Phase.MTP if self.client.config.mtp_depth else Phase.DECODE
                # A short credit cannot follow an uncertain speculative yield.
                if phase is Phase.MTP and pending and remaining < 4:
                    break
                output = min(4 if phase is Phase.MTP else 1, remaining)
                chunk = ()
            depth = self.client.config.mtp_depth if (phase is Phase.MTP or (phase is Phase.PREFILL and output)) else 0
            reserve = max(len(chunk) + output, depth + 1 if phase is Phase.MTP else 0) + max(0, depth - 1)
            if window.reservation_floor + reserve > self.client.config.max_context and not pending:
                # At the context end, authorize a target-only row and its history update.
                depth = 0
                if phase is Phase.MTP:
                    phase, output = Phase.DECODE, 1
                reserve = len(chunk) + output
            through = min(window.reservation_floor + reserve, self.client.config.max_context)
            if through < window.reservation_floor + reserve:
                if pending:
                    break
                raise RuntimeError("no context remains for the configured MTP state window")
            plan = window.grant(self.executable, phase, output, through, input_ids=tuple(chunk), draft_depth=depth)
            self.client.submit(plan)
            active.counters["authorized_steps"] += 1
            active.offset += len(chunk)
            if self.storage.prefix is not None and phase is Phase.PREFILL:
                break

    def complete(self, completion: Completion):
        active = self.active
        if active is None or completion.ticket.request != active.key:
            raise ValueError("completion does not belong to the active scheduler request")
        plans = {p.ticket.step_id: p for p in active.window.plans}
        active.counters["retired_credits"] += 1
        for key, value in (("proposed_drafts", completion.proposed_drafts),
                           ("verified_drafts", completion.verified_drafts),
                           ("sampler_accepted_drafts", completion.accepted_drafts)):
            if value is not None:
                active.counters[key] += value
        messages = []
        for result in self.storage.complete(active.lease, completion):
            req = active.request
            req.cache_handle = active.lease.handle
            plan = plans[result.ticket.step_id]
            active.counters["committed_steps"] += 1
            if result.tokens:
                tokens = torch.tensor(result.tokens, dtype=req.input_ids.dtype)
                message, committed = self._commit_tokens(req, tokens)
                req.stats.stamp_token(time.perf_counter())
                active.counters["committed_tokens"] += committed
                active.counters["committed_drafts"] += sum(result.draft_flags[:committed])
                if result.verified_drafts is not None:
                    req.stats.draft_n += result.verified_drafts
                    req.stats.draft_n_accepted += sum(result.draft_flags[:committed])
                if result.terminal_reason and not message.finished:
                    message.finished, message.finish_reason = True, result.terminal_reason
                if message.finished:
                    active.terminal_recorded = True
                    message.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
                    active.window.revoke()
                    active.terminal_message = message
                else:
                    messages.append(message)
            if result.after is not None:
                req.cached_len = result.after.computed_tokens
                req.device_len = max(req.input_ids.numel(), result.after.computed_tokens + 1)
                if plan.phase is Phase.PREFILL:
                    self.storage.publish(active.lease)
            if not active.window.revoked:
                self.decode_manager.filter_reqs([req])
        if active.window.revoked and not active.terminal_recorded:
            active.terminal_message = ErrorReplyMsg(active.request.uid, "executor retired the request without an output terminal")
            active.terminal_recorded = True
        messages.extend(self._drain_or_refill())
        return messages

    def cancel(self, uid):
        if self.active is None or self.active.request.uid != uid:
            return []
        active = self.active
        active.request.aborted = True
        active.window.revoke()
        if not active.terminal_recorded:
            active.terminal_message = ErrorReplyMsg(uid, "request aborted")
            active.terminal_recorded = True
        return self._drain_or_refill()

    def _drain_or_refill(self):
        active = self.active
        if active.window.revoked:
            if not active.lease.cancellation_sent:
                self.storage.revoke(active.lease)
            if active.window.drained:
                self.storage.release(active.lease)
                self.decode_manager.remove_req(active.request)
                self.active = None
                message = active.terminal_message
                if isinstance(message, DetokenizeMsg):
                    snapshot = self.client.inspect()
                    message.native_stats = dict(snapshot.pop("runtime", {}),
                        execution_backend="native-kernels", as_of="final_state_release",
                        completion_fence_drained=True, counts=dict(active.counters),
                        storage=self.storage.inspect(), executor=snapshot)
                return [message] if message is not None else []
        else:
            self.refill()
        return []
