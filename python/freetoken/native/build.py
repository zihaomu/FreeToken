"""Build and stage the bundled R9700 native runtime without a donor checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--backend", choices=["radiance-native","native-kernels"], default="radiance-native")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "kernel/csrc/radiance_native"
    batch = args.backend == "native-kernels"
    worker = "freetoken-batch-worker" if batch else "freetoken-native-worker"
    default_prefix = Path(__file__).parent / ("runtime-v2" if batch else "runtime")
    build, prefix = args.build_dir.resolve(), (args.prefix or default_prefix).resolve()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if not args.stage_only:
        compiler = shutil.which("g++-14")
        hip = Path(os.environ.get("ROCM_PATH", "/opt/rocm")) / "llvm/bin/clang++"
        if not compiler or not hip.is_file():
            parser.error("ROCm 10.1 and g++-14 are required by this runtime build")
        subprocess.run([
            "cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_CXX_COMPILER={compiler}",
            f"-DCMAKE_HIP_COMPILER={hip}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={hip.parents[2]}",
            "-DRAD_WITH_HIP=ON", "-DRAD_GPU_TARGETS=gfx1201",
            "-DRAD_WITH_FFMPEG=OFF", "-DRAD_BUILD_TESTS=OFF",
        ], check=True)
        subprocess.run(["cmake", "--build", str(build), "--target", worker,
                        "--parallel", str(args.jobs)], check=True)
    files = {
        "bin/" + worker: build / "bin" / worker,
        "share/radiance/kernels/libr4d.so": build / "radiance_home/kernels/libr4d.so",
        "share/radiance/kernels/libavx.so": build / "radiance_home/kernels/libavx.so",
        "share/radiance/kernels/spechub.so": build / "radiance_home/kernels/spechub.so",
        "share/radiance/quantizers/libquant.so": build / "radiance_home/quantizers/libquant.so",
        "licenses/radiance.LICENSE": source / "LICENSE",
        "licenses/radiance.NOTICE": source / "NOTICE",
        "licenses/llama.cpp.LICENSE": source / "vendor/llama/LICENSE",
        "licenses/yyjson.LICENSE": source / "vendor/yyjson/LICENSE",
        "licenses/cpp-httplib.LICENSE": source / "vendor/cpp-httplib/LICENSE",
        "licenses/rapidyaml.LICENSE": source / "vendor/rapidyaml/LICENSE.txt",
        "licenses/liquidai-d1.LICENSE": source / "core/server/LICENSE.lfm",
    }
    if not batch:
        files["share/radiance/architectures/qwen4exp_fp8.so"] = build / "radiance_home/architectures/qwen4exp_fp8.so"
    else:
        files["diagnostics/link.map"] = build / "freetoken-batch-worker.map"
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("runtime build is incomplete: " + ", ".join(missing))
    manifest = {"target": "gfx1201", "rocm": "10.1", "backend":args.backend, "files": {}}
    for name, path in files.items():
        target = prefix / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        manifest["files"][name] = {"sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                                    "bytes": target.stat().st_size}
    source_files = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(source.rglob("*")) if path.is_file()}
    manifest["source_files"] = source_files
    if batch:
        component = source.parent / "native_executor"
        manifest["freetoken_components"] = {
            str(p.relative_to(component)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(component.rglob("*")) if p.is_file()
        }
        manifest["framework_ownership"] = {"graph":"freetoken.models.qwen4_exp",
            "requests":"freetoken.scheduler", "prefix":"freetoken.kvcache", "placement":"freetoken.moe"}
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Native runtime staged at {prefix}")


if __name__ == "__main__":
    main()
