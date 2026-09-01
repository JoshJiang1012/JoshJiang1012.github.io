#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from lexa_asymcompute.models import HardwareSystem, WorkloadSpec
from lexa_asymcompute.moe import MoEHardwareSpec, MoEModelSpec, MoEPlacement, estimate_moe, optimize_moe
from lexa_asymcompute.optimizer import optimize


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    generic = optimize(
        WorkloadSpec.from_json(ROOT / "data/workloads/edge_pipeline_example.json"),
        HardwareSystem.from_json(ROOT / "data/hardware/desktop_heterogeneous_example.json"),
        step=0.01,
    )
    model = MoEModelSpec.from_json(ROOT / "data/workloads/sparse_moe_reference.json")
    hardware = MoEHardwareSpec.from_json(ROOT / "data/hardware/moe_three_tier_assumed.json")
    optimized = optimize_moe(model, hardware)
    cpu_baseline = estimate_moe(model, hardware, MoEPlacement(0.0, 1.0, 0.0))
    rows = [
        {
            "scenario": "generic-edge-pipeline",
            "classification": generic.classification,
            "total_ms": generic.total_ms,
            "units_per_second": generic.units_per_second,
            "placement": json.dumps(dict(generic.placement.shares), sort_keys=True),
        },
        {
            "scenario": "sparse-three-tier-cpu-baseline",
            "classification": cpu_baseline.classification,
            "total_ms": cpu_baseline.total_ms,
            "units_per_second": cpu_baseline.units_per_second,
            "placement": json.dumps({"gpu": 0.0, "ram": 1.0, "nvme": 0.0}, sort_keys=True),
        },
        {
            "scenario": "sparse-three-tier-optimized",
            "classification": optimized.classification,
            "total_ms": optimized.total_ms,
            "units_per_second": optimized.units_per_second,
            "placement": json.dumps({"gpu": optimized.placement.gpu, "ram": optimized.placement.ram, "nvme": optimized.placement.nvme}, sort_keys=True),
        },
    ]
    out_dir = ROOT / "data/analytical"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_scenarios.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out_dir / "reference_scenarios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
