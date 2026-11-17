from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import time
from pathlib import Path

from .config import NativeConfig
from .protocol import Kind, prompt_payload, recv_frame, send_frame


def inspect_model(worker: str, model: str) -> dict:
    result = subprocess.run([worker, "--inspect", model], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(result.stdout)


class NativeClient:
    def __init__(self, config: NativeConfig, work_dir: str, startup_timeout: float = 600) -> None:
        self.config = config
        self.path = Path(work_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "native.yaml").write_text(config.yaml())
        self.socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(startup_timeout)
        self.process = None
        self.log = (self.path / "native.stderr.log").open("wb")
        env = dict(os.environ, RADIANCE_HOME=config.runtime_home)
        # Torch preloads and package-local ROCm copies must not bind the AQL entry points.
        env.pop("LD_PRELOAD", None)
        env.pop("PYTHONPATH", None)
        env["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
        if config.visible_devices is not None:
            env["HIP_VISIBLE_DEVICES"] = config.visible_devices
            env["FREETOKEN_NATIVE_EXPECTED_PCIS"] = json.dumps(config.expected_pcis)
            env.pop("ROCR_VISIBLE_DEVICES", None)
            env.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            self.process = subprocess.Popen(
                [config.worker, "--config", str(self.path / "native.yaml"), "--ipc-fd", str(child_socket.fileno())],
                pass_fds=(child_socket.fileno(),), env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=self.log, start_new_session=True,
            )
            child_socket.close()
            kind, _, _, payload = self.receive()
            ready = json.loads(payload)
            if kind != Kind.READY:
                raise RuntimeError(f"native startup failed: {ready}")
            self.ready = ready
            (self.path / "native-ready.json").write_text(json.dumps(ready, indent=2) + "\n")
            self.socket.settimeout(5)
        except BaseException as exc:
            child_socket.close()
            self.close()
            if isinstance(exc, (EOFError, OSError)):
                tail = (self.path / "native.stderr.log").read_text(errors="replace")[-6000:]
                raise RuntimeError(f"native startup exited ({self.process.returncode if self.process else None}): {tail}") from exc
            raise

    def generate(self, uid: int, epoch: int, token_ids: list[int], metadata: dict) -> None:
        send_frame(self.socket, Kind.GENERATE, uid, epoch, prompt_payload(metadata, token_ids))

    def cancel(self, uid: int, epoch: int) -> None:
        send_frame(self.socket, Kind.CANCEL, uid, epoch)

    def receive(self) -> tuple[Kind, int, int, bytes]:
        return recv_frame(self.socket)

    def readable(self, timeout: float = 0) -> bool:
        return bool(select.select([self.socket], [], [], timeout)[0])

    def close(self) -> None:
        forced = False
        if self.process is not None and self.process.poll() is None:
            try:
                self.socket.settimeout(1)
                send_frame(self.socket, Kind.SHUTDOWN)
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                forced = True
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        (self.path / "native-exit.json").write_text(json.dumps({
            "returncode": self.process.returncode if self.process else None,
            "forced": forced, "closed_monotonic_ns": time.monotonic_ns(),
        }) + "\n")
        self.socket.close()
        self.log.close()

    def __enter__(self) -> NativeClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
