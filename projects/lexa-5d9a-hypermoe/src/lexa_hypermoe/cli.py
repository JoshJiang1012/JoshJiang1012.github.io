"""Command-line interface for the analytical model and router-trace toolkit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model import HardwareProfile, ModelSpec, optimize_placement
from .trace import (
    aggregate_cache_stats,
    cache_sweep,
    iter_jsonl,
    layer_cache_stats,
    layer_temporal_stats,
    trace_audit,
)


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


def _trace_events(args: argparse.Namespace):
    return tuple(
        iter_jsonl(
            args.trace,
            phase=getattr(args, "phase", None),
            strict_privacy=not getattr(args, "allow_unknown_fields", False),
        )
    )


def _analyze_trace(args: argparse.Namespace) -> int:
    events = _trace_events(args)
    stats = layer_cache_stats(
        events,
        cached_experts_per_layer=args.cached_experts_per_layer,
    )
    temporal = layer_temporal_stats(events)
    payload = {
        "trace": str(Path(args.trace).resolve()),
        "phase": args.phase,
        "cached_experts_per_layer": args.cached_experts_per_layer,
        "aggregate": aggregate_cache_stats(stats),
        "layers": [item.to_dict() for item in stats],
        "temporal_locality": [item.to_dict() for item in temporal],
        "classification": "observed_router_metadata_if_collected_from_runtime",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _audit_trace(args: argparse.Namespace) -> int:
    payload = trace_audit(
        args.trace,
        strict_privacy=not args.allow_unknown_fields,
    ).to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _sweep_trace(args: argparse.Namespace) -> int:
    sizes = [int(item.strip()) for item in args.cache_sizes.split(",") if item.strip()]
    if not sizes:
        raise ValueError("cache_sizes must contain at least one integer")
    events = _trace_events(args)
    payload = {
        "trace": str(Path(args.trace).resolve()),
        "phase": args.phase,
        "cache_sweep": list(cache_sweep(events, cache_sizes=sizes)),
        "classification": "observed_router_metadata_if_collected_from_runtime",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _add_trace_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument(
        "--allow-unknown-fields",
        action="store_true",
        help="disable the fail-closed privacy field allowlist",
    )


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

    trace = sub.add_parser("trace", help="analyze one router JSONL trace")
    _add_trace_common(trace)
    trace.add_argument("--cached-experts-per-layer", type=int, required=True)
    trace.set_defaults(handler=_analyze_trace)

    audit = sub.add_parser("trace-audit", help="validate privacy and summarize a trace")
    audit.add_argument("--trace", required=True)
    audit.add_argument("--allow-unknown-fields", action="store_true")
    audit.set_defaults(handler=_audit_trace)

    sweep = sub.add_parser("trace-sweep", help="compare several per-layer cache sizes")
    _add_trace_common(sweep)
    sweep.add_argument("--cache-sizes", default="4,8,12,16,18,20,24,32")
    sweep.set_defaults(handler=_sweep_trace)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
