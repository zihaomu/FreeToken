"""GPU-independent contracts shared by scheduling and device executors."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .model_plan import ModelPlan


def checked_int(value: int, name: str, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


class Phase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    MTP = "mtp"


@dataclass(frozen=True)
class RequestKey:
    uid: int
    epoch: int

    def __post_init__(self):
        checked_int(self.uid, "request uid", maximum=(1 << 64) - 1)
        checked_int(self.epoch, "request epoch", minimum=1, maximum=(1 << 64) - 1)


@dataclass(frozen=True)
class ResourceHandle:
    worker_generation: int
    model_generation: int
    resource_id: int
    version: int

    def __post_init__(self):
        for name in ("worker_generation", "model_generation", "resource_id", "version"):
            checked_int(getattr(self, name), name, minimum=1)


@dataclass(frozen=True)
class StateCursor:
    """The sampled last token may not yet have entered KV or recurrent state."""

    committed_tokens: int
    computed_tokens: int
    pending_token: int | None = None

    def __post_init__(self):
        checked_int(self.committed_tokens, "committed tokens")
        checked_int(self.computed_tokens, "computed tokens")
        difference = self.committed_tokens - self.computed_tokens
        if difference not in (0, 1) or (difference == 1) != (self.pending_token is not None):
            raise ValueError("state cursor must distinguish the uncomputed sampled token")
        if self.pending_token is not None:
            checked_int(self.pending_token, "pending token", maximum=(1 << 31) - 1)


@dataclass(frozen=True)
class StepTicket:
    request: RequestKey
    worker_generation: int
    credit_epoch: int
    step_id: int

    def __post_init__(self):
        for name in ("worker_generation", "credit_epoch", "step_id"):
            checked_int(getattr(self, name), name, minimum=1)


@dataclass(frozen=True)
class StepPlan:
    ticket: StepTicket
    executable: ResourceHandle
    state: ResourceHandle
    phase: Phase
    max_committed: int
    reserved_through: int
    input_ids: tuple[int, ...] = ()
    predecessor: StepTicket | None = None
    draft_depth: int = 0

    def __post_init__(self):
        if not isinstance(self.phase, Phase):
            raise ValueError("unknown step phase")
        for handle in (self.executable, self.state):
            if handle.worker_generation != self.ticket.worker_generation:
                raise ValueError("resource belongs to a different worker generation")
        if self.executable.model_generation != self.state.model_generation:
            raise ValueError("executable and state belong to different model generations")
        checked_int(self.reserved_through, "reserved frontier")
        checked_int(self.draft_depth, "draft depth")
        checked_int(self.max_committed, "maximum committed tokens", maximum=4)
        if type(self.input_ids) is not tuple:
            raise ValueError("input ids must be immutable")
        for token in self.input_ids:
            checked_int(token, "input token", maximum=(1 << 31) - 1)
        if self.phase is Phase.PREFILL:
            if not self.input_ids or self.draft_depth not in (0, 3) or self.max_committed > 1:
                raise ValueError("prefill requires prompt tokens and at most one sampled token")
        else:
            if self.input_ids or self.max_committed < 1:
                raise ValueError("decode consumes the state's pending token")
            if self.phase is Phase.DECODE and (self.draft_depth or self.max_committed != 1):
                raise ValueError("ordinary decode commits at most one token")
            if self.phase is Phase.MTP and (self.draft_depth != 3 or self.max_committed > 4):
                raise ValueError("this executor supports depth-three MTP")
        if self.predecessor is not None:
            p, t = self.predecessor, self.ticket
            if (p.request, p.worker_generation, p.credit_epoch) != (
                t.request, t.worker_generation, t.credit_epoch
            ) or p.step_id >= t.step_id:
                raise ValueError("invalid predecessor ticket")

    @property
    def reservation_tokens(self) -> int:
        trunk = max(len(self.input_ids) + self.max_committed,
                    self.draft_depth + 1 if self.phase is Phase.MTP else 0)
        # The head can write D-1 positions beyond the last verified trunk row.
        return trunk + max(0, self.draft_depth - 1)


@dataclass(frozen=True)
class Completion:
    ticket: StepTicket
    state: ResourceHandle
    before: StateCursor
    after: StateCursor | None
    tokens: tuple[int, ...] = ()
    draft_flags: tuple[bool, ...] = ()
    cancelled: bool = False
    terminal_frontier: int | None = None
    terminal_reason: str | None = None
    retired_computed_tokens: int | None = None
    proposed_drafts: int | None = None
    verified_drafts: int | None = None
    accepted_drafts: int | None = None

    def __post_init__(self):
        for name in ("proposed_drafts", "verified_drafts", "accepted_drafts"):
            value = getattr(self, name)
            if value is not None:
                checked_int(value, name, maximum=3)
        if self.accepted_drafts is not None and self.verified_drafts is not None and self.accepted_drafts > self.verified_drafts:
            raise ValueError("accepted drafts exceed verified drafts")
        if type(self.tokens) is not tuple or type(self.draft_flags) is not tuple:
            raise ValueError("completion token arrays must be immutable")
        if len(self.tokens) != len(self.draft_flags) or any(type(x) is not bool for x in self.draft_flags):
            raise ValueError("one committed-draft flag is required per token")
        for token in self.tokens:
            checked_int(token, "committed token", maximum=(1 << 31) - 1)
        if self.state.worker_generation != self.ticket.worker_generation:
            raise ValueError("completion belongs to a different worker")
        if type(self.cancelled) is not bool or (self.cancelled and self.tokens):
            raise ValueError("cancelled completions cannot publish tokens")
        terminal = self.terminal_reason is not None
        if terminal:
            if self.terminal_reason not in ("stop", "length", "cancelled", "error", "tool_calls"):
                raise ValueError("unknown terminal reason")
            if self.after is not None or self.terminal_frontier is None:
                raise ValueError("terminal state is retired, not a reusable cursor")
            checked_int(self.terminal_frontier, "terminal frontier")
            if self.retired_computed_tokens is not None:
                checked_int(self.retired_computed_tokens, "retired physical frontier")
        elif self.after is None or self.terminal_frontier is not None or self.retired_computed_tokens is not None:
            raise ValueError("live completion requires a reusable state cursor")


class ExecutionWindow:
    """Bounded authorization; cancellation revokes credits but does not free live storage."""

    def __init__(self, request: RequestKey, worker_generation: int, state: ResourceHandle,
                 cursor: StateCursor, *, limit: int = 1):
        checked_int(limit, "credit limit", minimum=1, maximum=2)
        checked_int(worker_generation, "worker generation", minimum=1)
        if state.worker_generation != worker_generation:
            raise ValueError("state belongs to a different worker")
        self.request, self.worker_generation = request, worker_generation
        self.state, self.cursor, self.limit = state, cursor, limit
        self.credit_epoch = 1
        self.revoked = False
        self.terminal_frontier: int | None = None
        self._next = 1
        self._pending: dict[int, StepPlan] = {}
        self._completed: dict[int, Completion] = {}

    @property
    def pending(self) -> tuple[StepTicket, ...]:
        return tuple(p.ticket for p in self._pending.values())

    @property
    def plans(self) -> tuple[StepPlan, ...]:
        return tuple(self._pending.values())

    @property
    def drained(self) -> bool:
        return not self._pending

    @property
    def reservation_floor(self) -> int:
        return self.cursor.committed_tokens + sum(
            p.reservation_tokens
            for p in self._pending.values()
        )

    def grant(self, executable: ResourceHandle, phase: Phase, max_committed: int,
              reserved_through: int, *, input_ids: tuple[int, ...] = (),
              draft_depth: int = 0) -> StepPlan:
        if self.revoked or len(self._pending) >= self.limit:
            raise RuntimeError("execution credits are revoked or exhausted")
        predecessor = next(reversed(self._pending.values())).ticket if self._pending else None
        if phase is Phase.PREFILL and self.cursor.pending_token is not None:
            raise ValueError("prefill cannot overwrite an uncomputed sampled token")
        if phase is not Phase.PREFILL and predecessor is None and self.cursor.pending_token is None:
            raise ValueError("decode needs a sampled token or an authorized predecessor")
        ticket = StepTicket(self.request, self.worker_generation, self.credit_epoch, self._next)
        plan = StepPlan(ticket, executable, self.state, phase, max_committed,
                        reserved_through, input_ids, predecessor, draft_depth)
        if reserved_through < self.reservation_floor + plan.reservation_tokens:
            raise ValueError("credit has insufficient worst-case state reservation")
        self._pending[ticket.step_id] = plan
        self._next += 1
        return plan

    def revoke(self) -> int:
        if not self.revoked:
            self.credit_epoch += 1
            self.revoked = True
        return self.credit_epoch

    def complete(self, result: Completion) -> tuple[Completion, ...]:
        plan = self._pending.get(result.ticket.step_id)
        if plan is None or result.ticket != plan.ticket or result.ticket.step_id in self._completed:
            raise ValueError("unknown, duplicate or stale completion")
        if (result.state.worker_generation, result.state.model_generation, result.state.resource_id) != (
            plan.state.worker_generation, plan.state.model_generation, plan.state.resource_id
        ):
            raise ValueError("completion changed its state allocation")
        if result.state.version <= plan.state.version:
            raise ValueError("completion did not advance state version")
        frontier = result.after.committed_tokens if result.after is not None else result.terminal_frontier
        if len(result.tokens) > plan.max_committed or frontier > plan.reserved_through:
            raise ValueError("completion exceeded its authorized bounds")
        if result.retired_computed_tokens is not None and result.retired_computed_tokens > plan.reserved_through:
            raise ValueError("terminal computation exceeded its physical state reservation")
        if plan.phase is not Phase.MTP and any(result.draft_flags):
            raise ValueError("non-MTP execution cannot accept draft tokens")
        self._completed[result.ticket.step_id] = result
        published = []
        while self._pending:
            step = next(iter(self._pending))
            if step not in self._completed:
                break
            current = self._completed[step]
            planned = self._pending[step]
            if not self.revoked:
                if current.cancelled:
                    self.revoke()
                else:
                    prompt = len(planned.input_ids) if planned.phase is Phase.PREFILL else 0
                    frontier = (current.after.committed_tokens if current.after is not None
                                else current.terminal_frontier)
                    if current.before != self.cursor or (
                        frontier !=
                        current.before.committed_tokens + prompt + len(current.tokens)
                    ) or current.state.version <= self.state.version:
                        raise ValueError("completion does not continue the committed state frontier")
                    self.state = current.state
                    if current.after is None:
                        self.terminal_frontier = frontier
                        self.revoke()
                    else:
                        self.cursor = current.after
                    published.append(current)
            del self._pending[step]
            del self._completed[step]
        return tuple(published)


class BatchExecutor(Protocol):
    def compile(self, plan: ModelPlan) -> ResourceHandle: ...
    def submit(self, plan: StepPlan) -> None: ...
    def poll(self, timeout: float = 0) -> Completion | None: ...
    def cancel(self, request: RequestKey, credit_epoch: int) -> None: ...
    def release(self, handle: ResourceHandle) -> None: ...
