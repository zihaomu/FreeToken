"""Versioned client for FreeToken bounded batches and state-storage commands."""
from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from freetoken.engine.backend import Completion, RequestKey, ResourceHandle, StateCursor, StepPlan
from .config import NativeConfig
from .executor_protocol import (
    Kind, decode, decode_completion, encode, encode_step, recv_frame, send_frame,
)


class BatchExecutorClient:
    def __init__(self, config: NativeConfig, work_dir: str | Path, *,
                 credit_window: int = 1, startup_timeout: float = 600, composition=None):
        if credit_window not in (1, 2):
            raise ValueError("executor credit window must be one or two")
        self.composition = composition
        if composition is not None and (
            len(composition.ranks) != config.tensor_parallel or composition.mtp_depth != config.mtp_depth
            or config.kv_dtype != "fp8"
        ):
            raise ValueError("model plan differs from runtime TP/MTP configuration")
        self.config = config
        self.path = Path(work_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self._completions: deque[Completion] = deque()
        self._request: RequestKey | None = None
        self.process = None
        self._forced = False
        self._log = (self.path / "executor.stderr.log").open("wb")
        self.socket, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(startup_timeout)
        (self.path / "executor.yaml").write_text(config.yaml())
        env = dict(os.environ, RADIANCE_HOME=config.runtime_home, LD_LIBRARY_PATH="/opt/rocm/lib")
        env.pop("LD_PRELOAD", None)
        env.pop("PYTHONPATH", None)
        if config.visible_devices is not None:
            env["HIP_VISIBLE_DEVICES"] = config.visible_devices
            env["FREETOKEN_NATIVE_EXPECTED_PCIS"] = json.dumps(config.expected_pcis)
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env.pop("ROCR_VISIBLE_DEVICES", None)
        plan_args = []
        if composition is not None:
            from freetoken.moe.residency import ExpertPlacementPlan
            plan_path = self.path / "composition.json"
            plan_path.write_bytes(composition.to_bytes())
            placement_path = self.path / "placement.json"
            placement_path.write_bytes(ExpertPlacementPlan.from_runtime(config,composition).to_bytes())
            plan_args = ["--model-plan", str(plan_path),"--placement-plan",str(placement_path)]
        try:
            self.process = subprocess.Popen(
                [config.worker, "--config", str(self.path / "executor.yaml"),
                 "--ipc-fd", str(child.fileno()), "--credit-window", str(credit_window), *plan_args],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=self._log,
                stderr=self._log, env=env, start_new_session=True,
            )
            child.close()
            kind, uid, epoch, payload = recv_frame(self.socket)
            self.ready = decode(payload)
            if kind is not Kind.READY or (uid, epoch) != (0, 0) or not (self.ready.get("bridge_only") or self.ready.get("batch_executor")):
                raise RuntimeError("unexpected bounded-step executor startup")
            self.worker_generation = self.ready["worker_generation"]
            (self.path / "executor-ready.json").write_text(json.dumps(self.ready, indent=2) + "\n")
            self.socket.settimeout(30)
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        kind, uid, epoch, payload = recv_frame(self.socket)
        if kind is Kind.ERROR:
            raise RuntimeError("executor: " + decode(payload)["error"])
        if kind is Kind.COMPLETE:
            result = decode_completion(payload)
            if (uid, epoch) != (result.ticket.request.uid, result.ticket.request.epoch):
                raise RuntimeError("executor completion frame identity mismatch")
            return kind, result
        return kind, (uid, epoch, decode(payload))

    def _rpc(self, kind: Kind, value: dict, request: RequestKey | None = None):
        uid, epoch = (request.uid, request.epoch) if request is not None else (0, 0)
        send_frame(self.socket, kind, uid, epoch, encode(value))
        while True:
            actual, payload = self._receive()
            if actual is Kind.COMPLETE:
                self._completions.append(payload)
                continue
            ru, re, reply = payload
            if (actual, ru, re) != (kind, uid, epoch):
                raise RuntimeError("executor RPC response does not match command")
            return reply

    def compile_composition(self) -> ResourceHandle:
        if self.composition is None:
            raise RuntimeError("no FreeToken composition supplied")
        reply = self._rpc(Kind.COMPILE, {"composition": json.loads(self.composition.to_bytes())})
        return ResourceHandle(**reply["executable"])

    def compile_reference(self) -> ResourceHandle:
        if self.composition is not None:
            raise RuntimeError("the worker loaded a FreeToken composition")
        reply = self._rpc(Kind.COMPILE, {"reference_program": True})
        return ResourceHandle(**reply["executable"])

    def reserve_reference(self, request: RequestKey, input_ids: list[int], *,
                          max_tokens: int, temperature: float = 0,
                          ignore_eos: bool = False, top_k: int = 0, top_p: float = 1.0,
                          frequency_penalty: float = 0, presence_penalty: float = 0) -> tuple[ResourceHandle, StateCursor]:
        if self._request is not None:
            raise RuntimeError("a executor state bundle is already active")
        reply = self._rpc(Kind.RESERVE, {"input_ids": input_ids, "max_tokens": max_tokens,
                         "temperature": temperature, "ignore_eos": ignore_eos,
                         "top_k": top_k, "top_p": top_p, "frequency_penalty": frequency_penalty,
                         "presence_penalty": presence_penalty}, request)
        self._request = request
        return ResourceHandle(**reply["state"]), StateCursor(**reply["cursor"])

    def submit(self, plan: StepPlan) -> None:
        if plan.ticket.request != self._request:
            raise ValueError("step does not belong to the reserved request")
        send_frame(self.socket, Kind.SUBMIT, self._request.uid, self._request.epoch, encode_step(plan))

    def poll(self, timeout: float = 0) -> Completion | None:
        if self._completions:
            return self._completions.popleft()
        if not select.select([self.socket], [], [], timeout)[0]:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"executor exited: {self.process.returncode}")
            return None
        kind, result = self._receive()
        if kind is not Kind.COMPLETE:
            raise RuntimeError("unexpected executor event")
        return result

    def cancel(self, request: RequestKey, credit_epoch: int) -> None:
        if request != self._request or credit_epoch < 2:
            raise ValueError("invalid executor revocation")
        self._rpc(Kind.REVOKE, {"credit_epoch": credit_epoch}, request)

    def release(self) -> None:
        if self._request is None:
            raise RuntimeError("no executor state allocation")
        self._rpc(Kind.RELEASE, {}, self._request)
        self._request = None

    def inspect(self) -> dict:
        return self._rpc(Kind.INSPECT, {})

    def snapshot(self, state, parent=None):
        reply=self._rpc(Kind.SNAPSHOT, {"state":asdict(state),
            "parent":asdict(parent) if parent is not None else None}, self._request)
        return ResourceHandle(**reply["snapshot"])

    def restore(self, state, snapshot):
        reply=self._rpc(Kind.RESTORE, {"state":asdict(state),"snapshot":asdict(snapshot)},self._request)
        return ResourceHandle(**reply["state"]),StateCursor(**reply["cursor"])

    def drop_snapshot(self, snapshot):
        self._rpc(Kind.DROP_SNAPSHOT, {"snapshot":asdict(snapshot)})

    def close(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                send_frame(self.socket, Kind.SHUTDOWN)
                process.wait(timeout=15)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self._forced = True
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        try:
            self.socket.close()
        finally:
            self._log.close()
        (self.path / "executor-exit.json").write_text(json.dumps({
            "returncode": process.returncode if process is not None else None,
            "forced": self._forced,
        }, indent=2) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# Historical intermediate experiments import this name.
ExecutorBridgeClient = BatchExecutorClient
