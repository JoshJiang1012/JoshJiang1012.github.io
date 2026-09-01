#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time


def _memory_copy(size: int, repeats: int) -> dict[str, float]:
    source = bytearray(os.urandom(size))
    target = bytearray(size)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        target[:] = source
        elapsed = time.perf_counter() - start
        samples.append(size / elapsed / 1e9)
    return {"median_gbps": statistics.median(samples), "min_gbps": min(samples), "max_gbps": max(samples)}


def _sha256(size: int, repeats: int) -> dict[str, float]:
    payload = os.urandom(size)
    samples = []
    digest = ""
    for _ in range(repeats):
        start = time.perf_counter()
        digest = hashlib.sha256(payload).hexdigest()
        elapsed = time.perf_counter() - start
        samples.append(size / elapsed / 1e9)
    return {"median_gbps": statistics.median(samples), "digest_prefix": digest[:16]}


def _file_io(size: int, repeats: int, directory: Path) -> dict[str, float]:
    payload = os.urandom(size)
    write_rates, read_rates = [], []
    for _ in range(repeats):
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as handle:
            path = Path(handle.name)
            start = time.perf_counter()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            write_rates.append(size / (time.perf_counter() - start) / 1e9)
        start = time.perf_counter()
        read_back = path.read_bytes()
        read_rates.append(len(read_back) / (time.perf_counter() - start) / 1e9)
        path.unlink()
    return {"write_median_gbps": statistics.median(write_rates), "read_median_gbps": statistics.median(read_rates)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size-mib", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--io-directory", default="/mnt/data")
    args = parser.parse_args()
    size = args.size_mib * 1024 * 1024
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    payload = {
        "schema_version": "1.0",
        "classification": "observed_measurement",
        "measured": True,
        "environment": "isolated-linux-virtual-environment",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "visible_memory_bytes": memory_bytes,
            "nvidia_gpu_visible": Path("/proc/driver/nvidia/gpus").exists(),
        },
        "parameters": {"size_bytes": size, "repeats": args.repeats},
        "measurements": {
            "python_bytearray_copy": _memory_copy(size, args.repeats),
            "sha256": _sha256(size, args.repeats),
            "filesystem": _file_io(size, max(2, min(args.repeats, 3)), Path(args.io_directory)),
        },
        "notes": "These measurements characterize this VM and the Python harness, not a target desktop GPU or model runtime."
    }
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
