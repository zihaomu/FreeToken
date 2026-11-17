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
    parser.add_argument("--prefix", type=Path, default=Path(__file__).parent / "runtime")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "kernel/csrc/radiance_native"
    build, prefix = args.build_dir.resolve(), args.prefix.resolve()
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
        subprocess.run(["cmake", "--build", str(build), "--target", "freetoken-native-worker",
                        "--parallel", str(args.jobs)], check=True)
    files = {
        "bin/freetoken-native-worker": build / "bin/freetoken-native-worker",
        "share/radiance/kernels/libr4d.so": build / "radiance_home/kernels/libr4d.so",
        "share/radiance/kernels/libavx.so": build / "radiance_home/kernels/libavx.so",
        "share/radiance/kernels/spechub.so": build / "radiance_home/kernels/spechub.so",
        "share/radiance/architectures/qwen4exp_fp8.so": build / "radiance_home/architectures/qwen4exp_fp8.so",
        "share/radiance/quantizers/libquant.so": build / "radiance_home/quantizers/libquant.so",
        "licenses/radiance.LICENSE": source / "LICENSE",
        "licenses/radiance.NOTICE": source / "NOTICE",
        "licenses/llama.cpp.LICENSE": source / "vendor/llama/LICENSE",
        "licenses/yyjson.LICENSE": source / "vendor/yyjson/LICENSE",
        "licenses/cpp-httplib.LICENSE": source / "vendor/cpp-httplib/LICENSE",
        "licenses/rapidyaml.LICENSE": source / "vendor/rapidyaml/LICENSE.txt",
        "licenses/liquidai-d1.LICENSE": source / "core/server/LICENSE.lfm",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        parser.error("runtime build is incomplete: " + ", ".join(missing))
    manifest = {"target": "gfx1201", "rocm": "10.1", "files": {}}
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
    (prefix / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Native runtime staged at {prefix}")


if __name__ == "__main__":
    main()
