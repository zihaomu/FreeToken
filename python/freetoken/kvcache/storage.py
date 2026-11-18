"""Resource-handle storage adapter; tensors stay in the device worker's one pool."""
from __future__ import annotations

from dataclasses import dataclass, replace

from freetoken.engine.backend import ExecutionWindow, RequestKey, ResourceHandle
from .base import BaseCacheHandle


def _compare_cpu_keys(a, b):
    if not a.is_cpu or not b.is_cpu:
        raise TypeError("prefix directory keys must stay on the CPU")
    n = min(a.numel(), b.numel())
    differences = (a[:n] != b[:n]).nonzero()
    return int(differences[0, 0]) if differences.numel() else n


@dataclass(frozen=True)
class StorageCacheHandle(BaseCacheHandle):
    resource: ResourceHandle
    retired: bool = False

    def get_matched_indices(self):
        raise TypeError("opaque state storage requires a storage adapter, not Torch KV indices")


@dataclass
class StateLease:
    request: RequestKey
    handle: StorageCacheHandle
    window: ExecutionWindow
    cancellation_sent: bool = False
    prefix_node: object = None
    snapshot_parent: ResourceHandle | None = None
    prompt: tuple[int, ...] = ()


class BatchStateStorage:
    def __init__(self, client, *, credit_window=2):
        self.client = client
        self.credit_window = credit_window
        self.leases: dict[RequestKey, StateLease] = {}
        self.prefix = None
        self.snapshots = {}
        self._snapshot_id = 0
        self.prefix_hits = self.prefix_tokens = self.prefix_publications = 0
        if client.config.prefix_cache:
            import torch
            from .hybrid_radix_cache import HybridRadixCache
            self.prefix = HybridRadixCache(torch.device("cpu"), page_size=4,
                snapshot_alignment=4, key_compare=_compare_cpu_keys)

    def _keys(self, prompt):
        import torch
        from freetoken.scheduler.core import prefill_chunk_size
        keys = []
        start = 0
        while start < len(prompt):
            end = start + prefill_chunk_size(start, len(prompt)-start,
                                             self.client.config.prefill_chunk, 4)
            for i in range(start, end):
                keys.extend((prompt[i], prompt[i+1] if self.client.config.mtp_depth and i+1<len(prompt) else -1,
                             start, end))
            start = end
        return torch.tensor(keys, dtype=torch.int64)

    def reserve(self, request, prompt, *, max_tokens, sampling_params):
        if self.leases:
            raise RuntimeError("this storage adapter grants one live B1 bundle")
        p = sampling_params
        state, cursor = self.client.reserve_reference(
            request, list(prompt), max_tokens=max_tokens, temperature=p.temperature,
            ignore_eos=p.ignore_eos, top_k=max(p.top_k, 0), top_p=p.top_p,
            frequency_penalty=p.frequency_penalty, presence_penalty=p.presence_penalty,
        )
        match = None
        parent = None
        if self.prefix is not None:
            match = self.prefix.match_prefix(self._keys(prompt)[:(len(prompt)-1)*4])
            if match.mamba_value is not None:
                parent = self.snapshots[match.mamba_value]
                self.prefix.inc_lock(match.node)
                try:
                    state, cursor = self.client.restore(state, parent)
                except BaseException:
                    self.prefix.dec_lock(match.node)
                    raise
                if cursor.computed_tokens != match.cached_len//4:
                    raise RuntimeError("physical restore disagrees with the prefix directory")
                self.prefix_hits += 1
                self.prefix_tokens += cursor.computed_tokens
        handle = StorageCacheHandle(resource=state, cached_len=cursor.computed_tokens)
        window = ExecutionWindow(request, self.client.worker_generation, state, cursor,
                                 limit=self.credit_window)
        lease = StateLease(request, handle, window, prefix_node=match.node if parent else None,
                           snapshot_parent=parent, prompt=tuple(prompt))
        self.leases[request] = lease
        return lease

    def complete(self, lease, completion):
        if self.leases.get(lease.request) is not lease:
            raise ValueError("unknown or retired state lease")
        ordered = lease.window.complete(completion)
        lease.handle = replace(lease.handle, resource=lease.window.state,
            cached_len=lease.window.cursor.computed_tokens,
            retired=lease.window.terminal_frontier is not None)
        return ordered

    def publish(self, lease):
        window = lease.window
        n = window.cursor.computed_tokens
        if self.prefix is None or not window.drained or window.revoked or n<=0 or n>=len(lease.prompt) or n%4:
            return
        keys = self._keys(lease.prompt)[:n*4]
        match = self.prefix.match_prefix(keys)
        if match.cached_len == len(keys):
            lease.snapshot_parent = self.snapshots[match.mamba_value]
            return
        while True:
            try:
                snapshot = self.client.snapshot(window.state, lease.snapshot_parent)
                break
            except RuntimeError as error:
                if str(error) != "executor: prefix_capacity_exhausted":
                    raise
                evicted = self.prefix.evict_mamba(1)
                if not evicted.mamba_slots:
                    return
                for sid in evicted.mamba_slots:
                    old = self.snapshots.pop(sid)
                    if old == lease.snapshot_parent:
                        # Shared host chunks remain held by descendants. A removed parent
                        # can no longer authorize sharing, so the next snapshot is complete.
                        lease.snapshot_parent = None
                    self.client.drop_snapshot(old)
        import torch
        self._snapshot_id += 1
        sid = self._snapshot_id
        _, exists = self.prefix.insert(keys, torch.arange(len(keys), dtype=torch.int32), sid)
        if exists:
            self.client.drop_snapshot(snapshot)
        else:
            self.snapshots[sid] = snapshot
            lease.snapshot_parent = snapshot
            self.prefix_publications += 1

    def flush(self):
        if self.leases:
            raise RuntimeError("prefix flush requires no live request lease")
        if self.prefix is not None:
            evicted = self.prefix.evict_full(self.prefix.full_evictable_size)
            for sid in evicted.mamba_slots:
                self.client.drop_snapshot(self.snapshots.pop(sid))
            if self.snapshots:
                raise RuntimeError("prefix tree left orphaned snapshot leases")

    def inspect(self):
        return {"owner":"freetoken", "live_leases":len(self.leases),
                "prefix_snapshots":len(self.snapshots), "prefix_hits":self.prefix_hits,
                "prefix_tokens":self.prefix_tokens, "prefix_publications":self.prefix_publications}

    def revoke(self, lease):
        if self.leases.get(lease.request) is not lease:
            raise ValueError("unknown state lease")
        lease.window.revoke()
        if not lease.cancellation_sent:
            self.client.cancel(lease.request, lease.window.credit_epoch)
            lease.cancellation_sent = True

    def release(self, lease):
        if self.leases.get(lease.request) is not lease or not lease.window.drained:
            raise RuntimeError("state release requires its final completion fence")
        if not lease.window.revoked:
            raise RuntimeError("live state cannot be released without revoking its credits")
        self.client.release()
        if lease.prefix_node is not None:
            self.prefix.dec_lock(lease.prefix_node)
        lease.handle = replace(lease.handle, retired=True)
        del self.leases[lease.request]
