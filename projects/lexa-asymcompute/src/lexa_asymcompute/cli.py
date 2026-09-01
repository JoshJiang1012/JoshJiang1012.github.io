from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from . import __version__
from .models import FractionalPlacement, HardwareSystem, WorkloadSpec, estimate
from .moe import MoEHardwareSpec, MoEModelSpec, moe_savings, optimize_moe
from .optimizer import ObjectiveWeights, optimize
from .trace import load_trace, temporal_top_n_holdout, validate_geometry


def _write(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _parse_shares(text: str) -> dict[str, float]:
    path = Path(text)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("placement must be a JSON object")
    return {str(key): float(value) for key, value in raw.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lexa-asymcompute",
        description="Model and optimize asymmetric compute across heterogeneous tiers.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    estimate_parser = sub.add_parser("estimate", help="estimate a user-supplied placement")
    estimate_parser.add_argument("--workload", required=True)
    estimate_parser.add_argument("--hardware", required=True)
    estimate_parser.add_argument("--placement", required=True, help='JSON object or path, e.g. {"gpu":0.7,"cpu":0.3}')

    optimize_parser = sub.add_parser("optimize", help="grid-search a feasible fractional placement")
    optimize_parser.add_argument("--workload", required=True)
    optimize_parser.add_argument("--hardware", required=True)
    optimize_parser.add_argument("--step", type=float, default=0.01)
    optimize_parser.add_argument("--latency-weight", type=float, default=1.0)
    optimize_parser.add_argument("--energy-weight", type=float, default=0.0)

    savings_parser = sub.add_parser("moe-savings", help="compute sparse-MoE analytical proxies")
    savings_parser.add_argument("--model", required=True)

    moe_parser = sub.add_parser("moe-optimize", help="optimize a GPU/RAM/NVMe sparse workload profile")
    moe_parser.add_argument("--model", required=True)
    moe_parser.add_argument("--hardware", required=True)
    moe_parser.add_argument("--step", type=float, default=0.0025)

    trace_parser = sub.add_parser("trace", help="validate and evaluate a privacy-safe routing trace")
    trace_parser.add_argument("--trace", required=True)
    trace_parser.add_argument("--layers", type=int, required=True)
    trace_parser.add_argument("--top-k", type=int, required=True)
    trace_parser.add_argument("--experts-per-layer", type=int, required=True)
    trace_parser.add_argument("--cache", type=int, default=18)
    trace_parser.add_argument("--calibration", type=float, default=0.70)

    sub.add_parser("formula-summary", help="print the core equations in machine-readable form")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "estimate":
            workload = WorkloadSpec.from_json(args.workload)
            hardware = HardwareSystem.from_json(args.hardware)
            result = estimate(workload, hardware, FractionalPlacement(_parse_shares(args.placement)))
            _write(result.to_dict())
        elif args.command == "optimize":
            workload = WorkloadSpec.from_json(args.workload)
            hardware = HardwareSystem.from_json(args.hardware)
            result = optimize(
                workload,
                hardware,
                step=args.step,
                weights=ObjectiveWeights(args.latency_weight, args.energy_weight),
            )
            _write(result.to_dict())
        elif args.command == "moe-savings":
            _write(moe_savings(MoEModelSpec.from_json(args.model)))
        elif args.command == "moe-optimize":
            result = optimize_moe(
                MoEModelSpec.from_json(args.model),
                MoEHardwareSpec.from_json(args.hardware),
                args.step,
            )
            _write(result.to_dict())
        elif args.command == "trace":
            events = load_trace(args.trace)
            geometry = validate_geometry(
                events,
                layers=args.layers,
                top_k=args.top_k,
                experts_per_layer=args.experts_per_layer,
            )
            if not geometry["valid"]:
                _write({"geometry": geometry})
                return 2
            _write(
                {
                    "geometry": geometry,
                    "holdout": temporal_top_n_holdout(
                        events,
                        cache_per_layer=args.cache,
                        calibration_fraction=args.calibration,
                    ),
                }
            )
        elif args.command == "formula-summary":
            _write(
                {
                    "device_service": "max(F_d/P_d, B_d/M_d) + latency_d",
                    "transfer": "S_ij/link_bandwidth + link_latency",
                    "critical_path": "serial + max(device_path_d) + sync",
                    "objective": "latency_weight*T + energy_weight*E",
                    "asymmetry_index": "1 - (sum(c)^2 / (n*sum(c^2)))",
                    "speedup": "T_baseline / T_optimized",
                }
            )
        else:  # pragma: no cover
            raise AssertionError("unreachable command")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0
