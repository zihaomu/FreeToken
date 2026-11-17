from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from freetoken.distributed import DistributedInfo
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
from freetoken.native.client import inspect_model
from freetoken.native.config import NativeConfig

from .args import ServerArgs


@dataclass(frozen=True)
class NativeServerArgs(ServerArgs):
    native_config: NativeConfig | None = None
    native_work_dir: str = ""
    native_descriptor: dict = field(default_factory=dict)

    @property
    def model_config(self):
        meta = self.native_descriptor.get("metadata", {})
        return SimpleNamespace(
            has_linear_attention=True, is_moe=True,
            num_experts=int(meta.get("n_expert", 0)),
            num_moe_layers=int(meta.get("n_layers", 0)),
        )

    @property
    def served_modalities(self):
        return frozenset()

    @property
    def active_encoders(self):
        return ()


def launch_native(argv: list[str], run_shell: bool = False, prog: str | None = None,
                  visibility: str = "", expected_pcis: str = "[]") -> None:
    from freetoken.launch import resolve_api_key
    from .api_server import run_api_server
    from .launch import _run_tokenize_worker
    from .supervisor import BackendHandle
    from freetoken.engine.native_backend import run_native_backend

    p = argparse.ArgumentParser(prog=prog or "ft serve")
    p.add_argument("--execution-backend", choices=["radiance-native"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer-path", required=True)
    runtime = Path(__file__).resolve().parents[1] / "native/runtime"
    p.add_argument("--native-worker", default=os.environ.get("FREETOKEN_NATIVE_WORKER", str(runtime / "bin/freetoken-native-worker")))
    p.add_argument("--native-home", default=os.environ.get("RADIANCE_HOME", str(runtime / "share/radiance")))
    p.add_argument("--native-work-dir", default=None)
    p.add_argument("--tensor-parallel-size", type=int, default=2)
    p.add_argument("--max-running-requests", type=int, choices=[1], default=1)
    p.add_argument("--max-seq-len-override", type=int, default=34816)
    p.add_argument("--native-preset", choices=["flashnext-speed"], default="flashnext-speed")
    p.add_argument("--native-mtp-depth", type=int, choices=[0, 3], default=3)
    p.add_argument("--native-kv-cache-dtype", choices=["bf16", "i8", "fp8"], default="fp8")
    p.add_argument("--native-wire", choices=["exact", "wht6"], default="exact")
    p.add_argument("--native-host-pool-mib", type=int, default=12288)
    p.add_argument("--native-headroom-mib", type=int, default=96)
    p.add_argument("--native-gpu-memory-mib", type=int, default=0)
    p.add_argument("--native-ngram-share-dir", default="")
    p.add_argument("--native-prefix-cache", action="store_true")
    p.add_argument("--native-prefix-cache-mib", type=int, default=0)
    p.add_argument("--max-extend-tokens", type=int, default=2048)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=1919)
    p.add_argument("--served-model-name", default="qwen38-native")
    p.add_argument("--api-key", default=None)
    p.add_argument("--enable-timings-report", action="store_true")
    p.add_argument("--enable-cache-report", action="store_true")
    p.add_argument("--tool-call-parser", default="qwen3_coder")
    p.add_argument("--reasoning-parser", default="qwen3")
    p.add_argument("--sampling-defaults", choices=["model", "none"], default="none")
    p.add_argument("--max-output-tokens", type=int, default=None)
    args = p.parse_args(argv)
    if not Path(args.tokenizer_path).is_dir():
        p.error("--tokenizer-path must name a local Hugging Face tokenizer directory")
    if args.api_key == "":
        p.error("--api-key is empty; omit it to serve without authentication")
    try:
        native = NativeConfig(
            args.model, args.native_worker, args.native_home, args.tensor_parallel_size,
            args.max_seq_len_override, args.max_extend_tokens, args.native_mtp_depth,
            args.native_kv_cache_dtype, args.native_wire, args.native_host_pool_mib,
            args.native_headroom_mib, args.native_prefix_cache, args.native_ngram_share_dir,
            visibility, tuple(json.loads(expected_pcis)), args.native_gpu_memory_mib,
            args.native_prefix_cache_mib,
        )
        descriptor = inspect_model(native.worker, native.model)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    work = args.native_work_dir or tempfile.mkdtemp(prefix="freetoken-native-")
    config = NativeServerArgs(
        model_path=args.tokenizer_path, tp_info=DistributedInfo(0, native.tensor_parallel),
        dtype=torch.bfloat16, native_config=native, native_work_dir=work,
        native_descriptor=descriptor, max_running_req=1,
        max_seq_len_override=native.max_context, max_extend_tokens=native.prefill_chunk,
        server_host=args.host, server_port=args.port, api_key=resolve_api_key(args.api_key),
        served_model_name=args.served_model_name, tool_call_parser=args.tool_call_parser,
        reasoning_parser=None if args.reasoning_parser == "off" else args.reasoning_parser,
        sampling_defaults=args.sampling_defaults, max_output_tokens=args.max_output_tokens,
        enable_timings_report=args.enable_timings_report, enable_cache_report=args.enable_cache_report,
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)), shell_mode=run_shell,
    )

    def start_backend():
        mp.set_start_method("spawn", force=True)
        ack = mp.Queue()
        processes = [mp.Process(target=run_native_backend, args=(config, ack), name="freetoken-native-adapter")]
        processes.append(mp.Process(target=_run_tokenize_worker, kwargs={
            "detach": run_shell, "tokenizer_path": config.model_path,
            "addr": config.zmq_detokenizer_addr, "backend_addr": config.zmq_backend_addr,
            "frontend_addr": config.zmq_frontend_addr, "local_bs": 1,
            "mm": config.mm, "create": config.tokenizer_create_addr, "ack_queue": ack,
        }, name="freetoken-native-tokenizer"))
        for process in processes:
            process.start()
        return BackendHandle(ack, processes, 2)

    run_api_server(config, start_backend, run_shell=run_shell)
