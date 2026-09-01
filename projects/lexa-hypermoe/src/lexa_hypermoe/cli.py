from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .model import HardwareSpec, ModelSpec, Placement, estimate, savings
from .optimizer import optimize
from .trace import holdout_topn, read_trace, validate_trace


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lexa-hypermoe")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    estimate_cmd = commands.add_parser("estimate", help="estimate one placement")
    estimate_cmd.add_argument("--model", required=True)
    estimate_cmd.add_argument("--hardware", required=True)
    estimate_cmd.add_argument("--gpu", type=float, required=True)
    estimate_cmd.add_argument("--ram", type=float, required=True)
    estimate_cmd.add_argument("--nvme", type=float, required=True)

    optimize_cmd = commands.add_parser("optimize", help="grid-search a placement")
    optimize_cmd.add_argument("--model", required=True)
    optimize_cmd.add_argument("--hardware", required=True)
    optimize_cmd.add_argument("--step", type=float, default=0.0025)
    optimize_cmd.add_argument("--gpu-cap", type=float, default=1.0)

    savings_cmd = commands.add_parser("savings", help="show analytical MoE savings")
    savings_cmd.add_argument("--model", required=True)

    trace_cmd = commands.add_parser("trace", help="evaluate a Router Trace holdout")
    trace_cmd.add_argument("--trace", required=True)
    trace_cmd.add_argument("--model", required=True)
    trace_cmd.add_argument("--cache", type=int, default=18)
    trace_cmd.add_argument("--calibration", type=float, default=0.70)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "estimate":
        model = ModelSpec.from_json(args.model)
        hardware = HardwareSpec.from_json(args.hardware)
        _print(estimate(model, hardware, Placement(args.gpu, args.ram, args.nvme)).to_dict())
    elif args.command == "optimize":
        model = ModelSpec.from_json(args.model)
        hardware = HardwareSpec.from_json(args.hardware)
        _print(optimize(model, hardware, step=args.step, gpu_fraction_cap=args.gpu_cap).to_dict())
    elif args.command == "savings":
        _print(savings(ModelSpec.from_json(args.model)))
    elif args.command == "trace":
        model = ModelSpec.from_json(args.model)
        records = validate_trace(
            read_trace(args.trace),
            layers=model.layers,
            experts_per_layer=model.experts_per_layer,
            active_experts=model.active_experts_per_token,
        )
        _print(
            holdout_topn(
                records,
                cached_experts_per_layer=args.cache,
                calibration_fraction=args.calibration,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
