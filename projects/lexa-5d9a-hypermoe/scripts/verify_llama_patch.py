#!/usr/bin/env python3
"""Verify or apply the pinned llama.cpp router-trace patch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/llama.cpp/d08c7872-router-trace.patch"
MANIFEST = ROOT / "patches/llama.cpp/manifest.json"


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-source", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-different-commit", action="store_true")
    args = parser.parse_args(argv)

    source = args.llama_source.expanduser().resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_commit = str(manifest["upstream_commit"])
    expected_patch_sha = str(manifest["patch_sha256"])
    actual_patch_sha = sha256(PATCH)
    if actual_patch_sha != expected_patch_sha:
        parser.error(
            f"patch digest mismatch: manifest={expected_patch_sha} actual={actual_patch_sha}"
        )

    if not (source / ".git").exists():
        parser.error("--llama-source must point to a Git checkout")

    head = run(["git", "rev-parse", "HEAD"], cwd=source)
    if head.returncode != 0:
        parser.error(head.stderr.strip() or "could not read llama.cpp commit")
    actual_commit = head.stdout.strip()
    if actual_commit != expected_commit and not args.allow_different_commit:
        parser.error(
            f"llama.cpp commit mismatch: expected {expected_commit}, got {actual_commit}"
        )

    openai_moe = source / "src/models/openai-moe.cpp"
    graph = source / "src/llama-graph.cpp"
    required_markers = {
        openai_moe: "LLM_FFN_SWIGLU_OAI_MOE",
        graph: 'cb(selected_experts, "ffn_moe_topk", il);',
    }
    for path, marker in required_markers.items():
        if not path.exists() or marker not in path.read_text(encoding="utf-8"):
            parser.error(f"upstream source marker missing: {path.relative_to(source)}: {marker}")

    command = ["git", "apply", "--check", str(PATCH)]
    checked = run(command, cwd=source)
    already_applied = (source / "examples/router-trace/router-trace.cpp").exists()
    if checked.returncode != 0 and not already_applied:
        sys.stderr.write(checked.stderr)
        return 1

    if args.apply and not already_applied:
        applied = run(["git", "apply", str(PATCH)], cwd=source)
        if applied.returncode != 0:
            sys.stderr.write(applied.stderr)
            return applied.returncode

    payload = {
        "ok": True,
        "source": str(source),
        "expected_commit": expected_commit,
        "actual_commit": actual_commit,
        "patch_sha256": actual_patch_sha,
        "already_applied": already_applied,
        "applied": bool(args.apply and not already_applied),
        "target": "llama-router-trace",
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
