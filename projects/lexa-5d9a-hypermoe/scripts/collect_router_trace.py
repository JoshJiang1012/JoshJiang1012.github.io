#!/usr/bin/env python3
"""Run the patched llama-router-trace tool and write a privacy-safe run manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable

from lexa_hypermoe.trace import trace_audit


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitized_command(command: Iterable[str], prompt_file: Path) -> list[str]:
    result: list[str] = []
    for item in command:
        result.append("<temporary-prompt-file>" if item == str(prompt_file) else item)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--domain", default="unspecified")
    parser.add_argument("--n-predict", type=int, default=2048)
    parser.add_argument("--gpu-layers", type=int, default=99)
    parser.add_argument(
        "--n-cpu-moe",
        type=int,
        default=0,
        help=(
            "keep MoE weights of the first N layers on the CPU; GPT-OSS-120B has "
            "36 layers, so 36 is the conservative 16GB-VRAM collection profile"
        ),
    )
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--trace-prefill", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    parser.add_argument("--model-sha256")
    parser.add_argument(
        "--hash-model",
        action="store_true",
        help="hash the complete model file; this can take significant time for 60+ GB weights",
    )
    args = parser.parse_args(argv)

    binary = args.binary.expanduser().resolve()
    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must be an executable file")
    if not model.is_file():
        parser.error("--model must be a file")
    if args.n_predict < 1 or args.ctx_size < 2:
        parser.error("n-predict and ctx-size must be positive")
    if args.gpu_layers < 0 or args.n_cpu_moe < 0:
        parser.error("gpu-layers and n-cpu-moe must be non-negative")

    if args.prompt_file is not None:
        prompt_text = args.prompt_file.expanduser().read_text(encoding="utf-8")
    else:
        prompt_text = str(args.prompt)
    if not prompt_text.strip():
        parser.error("prompt must not be empty")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(output.suffix + ".run.json")
    started = datetime.now(timezone.utc)

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="lexa-router-prompt-",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(prompt_text)
            temp_name = handle.name
        prompt_path = Path(temp_name)
        if os.name != "nt":
            os.chmod(prompt_path, 0o600)

        command = [
            str(binary),
            "-m",
            str(model),
            "-o",
            str(output),
            "-n",
            str(args.n_predict),
            "-ngl",
            str(args.gpu_layers),
            "--ctx-size",
            str(args.ctx_size),
            "--domain",
            args.domain,
            "--prompt-file",
            str(prompt_path),
            "--quiet",
        ]
        if args.n_cpu_moe:
            command.extend(["--n-cpu-moe", str(args.n_cpu_moe)])
        if args.trace_prefill:
            command.append("--trace-prefill")

        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=args.timeout_seconds,
            check=False,
        )
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)

    ended = datetime.now(timezone.utc)
    model_digest = args.model_sha256
    if args.hash_model:
        model_digest = file_sha256(model)

    manifest = {
        "schema_version": "1.0",
        "data_class": "observed_router_metadata_if_runtime_completed",
        "started_at": started.isoformat(),
        "completed_at": ended.isoformat(),
        "duration_seconds": (ended - started).total_seconds(),
        "return_code": completed.returncode,
        "binary": {
            "path": str(binary),
            "sha256": file_sha256(binary),
        },
        "model": {
            "path": str(model),
            "bytes": model.stat().st_size,
            "sha256": model_digest,
        },
        "trace": str(output),
        "domain": args.domain,
        "n_predict": args.n_predict,
        "gpu_layers": args.gpu_layers,
        "n_cpu_moe": args.n_cpu_moe,
        "ctx_size": args.ctx_size,
        "trace_prefill": args.trace_prefill,
        "prompt": {
            "sha256": hash_text(prompt_text),
            "utf8_bytes": len(prompt_text.encode("utf-8")),
            "stored": False,
        },
        "command": sanitized_command(command, prompt_path),
        "stderr_tail": completed.stderr[-4000:],
    }

    if output.exists() and output.stat().st_size:
        try:
            manifest["trace_audit"] = trace_audit(output).to_dict()
        except ValueError as exc:
            manifest["trace_audit_error"] = str(exc)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
