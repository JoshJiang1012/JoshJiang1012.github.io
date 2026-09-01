"""Command-line interface for the analytical model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import HardwareProfile, ModelSpec, optimize_placement
from .trace import aggregate_cache_stats, iter_jsonl, layer_cache_stats


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _model_from_json(path: str | Path) -> ModelSpec:
    raw = _load_json(path)
    allowed = ModelSpec.__dataclass_fields__.keys()
    return ModelSpec(**{key: raw[key] for key in allowed if key in raw})


def _hardware_from_json(path: str | Path) -> HardwareProfile:
    raw = _load_json(path)
    allowed = HardwareProfile.__dataclass_fields__.keys()
    return HardwareProfile(**{key: raw[key] for key in allowed if key in raw})


def _estimate(args: argparse.Namespace) -> int:
    spec = _model_from_json(args.model)
    hardware = _hardware_from_json(args.hardware)
    result = optimize_placement(
        spec,
        hardware,
        step=args.step,
        max_gpu_fraction=args.max_gpu_fraction,
        critical_miss_probability_per_layer=args.layer_miss_probability,
        critical_miss_penalty_ms=args.miss_penalty_ms,
    )
    payload = {
        "model": spec.to_dict(),
        "hardware": hardware.to_dict(spec),
        "estimate": result.to_dict(),
        "classification": "analytical_estimate_not_observed_benchmark",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _analyze_trace(args: argparse.Namespace) -> int:
    stats = layer_cache_stats(
        iter_jsonl(args.trace),
        cached_experts_per_layer=args.cached_experts_per_layer,
    )
    payload = {
        "trace": str(Path(args.trace).resolve()),
        "cached_experts_per_layer": args.cached_experts_per_layer,
        "aggregate": aggregate_cache_stats(stats),
        "layers": [item.to_dict() for item in stats],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexa-hypermoe")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate = sub.add_parser("estimate", help="estimate best tier placement")
    estimate.add_argument("--model", required=True)
    estimate.add_argument("--hardware", required=True)
    estimate.add_argument("--step", type=float, default=0.005)
    estimate.add_argument("--max-gpu-fraction", type=float, default=1.0)
    estimate.add_argument("--layer-miss-probability", type=float, default=0.0)
    estimate.add_argument("--miss-penalty-ms", type=float, default=0.0)
    estimate.set_defaults(handler=_estimate)

    trace = sub.add_parser("trace", help="analyze a router JSONL trace")
    trace.add_argument("--trace", required=True)
    trace.add_argument("--cached-experts-per-layer", type=int, required=True)
    trace.set_defaults(handler=_analyze_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser = build_parser()
        parser.error(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
