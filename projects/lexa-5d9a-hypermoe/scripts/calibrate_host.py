#!/usr/bin/env python3
"""Collect a conservative host profile without executing model inference.

The memory-copy benchmark is a lightweight calibration signal, not a substitute
for STREAM, llama-bench, or a fused MXFP4 kernel benchmark. No privileged access
is required. Optional NVMe measurement reads an existing file and never writes.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def memory_copy_gbps(size_mib: int, repeats: int) -> dict[str, float]:
    if size_mib < 16 or repeats < 1:
        raise ValueError("size_mib must be >= 16 and repeats >= 1")
    size = size_mib * 1024 * 1024
    source = bytearray(os.urandom(min(size, 1024 * 1024)))
    source *= size // len(source)
    source.extend(b"\0" * (size - len(source)))
    destination = bytearray(size)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        destination[:] = source
        elapsed = time.perf_counter() - started
        samples.append(size / elapsed / 1e9)
    samples.sort()
    return {
        "minimum_gbps": samples[0],
        "median_gbps": samples[len(samples) // 2],
        "maximum_gbps": samples[-1],
        "size_mib": float(size_mib),
        "repeats": float(repeats),
    }


def read_file_gbps(path: Path, chunk_mib: int = 16) -> dict[str, float | int | str]:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError("NVMe calibration file must be non-empty")
    chunk = chunk_mib * 1024 * 1024
    total = 0
    started = time.perf_counter()
    with path.open("rb", buffering=0) as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            total += len(data)
    elapsed = time.perf_counter() - started
    return {
        "path": str(path.resolve()),
        "bytes_read": total,
        "elapsed_seconds": elapsed,
        "sequential_read_gbps": total / elapsed / 1e9,
        "warning": "OS page cache may inflate this result; use a dedicated tool for publication.",
    }


def nvidia_smi() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False}
    fields = [
        "name",
        "driver_version",
        "memory.total",
        "memory.used",
        "memory.free",
        "temperature.gpu",
        "power.limit",
    ]
    command = [
        executable,
        "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return {"available": True, "error": completed.stderr.strip()}
    rows = []
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        rows.append(dict(zip(fields, values, strict=False)))
    return {"available": True, "gpus": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-size-mib", type=int, default=256)
    parser.add_argument("--memory-repeats", type=int, default=5)
    parser.add_argument("--nvme-file", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload: dict[str, object] = {
        "classification": "observed_host_calibration_not_model_benchmark",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "memory_copy": memory_copy_gbps(args.memory_size_mib, args.memory_repeats),
        "nvidia_smi": nvidia_smi(),
        "timestamp_unix": time.time(),
    }
    if args.nvme_file is not None:
        payload["nvme_read"] = read_file_gbps(args.nvme_file)

    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
