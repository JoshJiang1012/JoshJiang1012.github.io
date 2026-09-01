#!/usr/bin/env python3
"""Generate all analytical CSV/JSON artifacts deterministically."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lexa_hypermoe.model import (  # noqa: E402
    HardwareProfile,
    ModelSpec,
    all_active_experts_hit_approx,
    layer_budget_ms,
    max_layer_miss_probability,
    optimize_placement,
    prefetch_horizon_layers,
    zipf_probability_mass,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows generated for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    spec = ModelSpec()
    model_path = ROOT / "data/model/gpt_oss_120b.json"
    model_path.write_text(
        json.dumps(
            {
                **spec.to_dict(),
                "source_class": "official_config_plus_derived_estimates",
                "source_notes": [
                    "Architecture constants come from openai/gpt-oss-120b config.json.",
                    "5.1B active parameters comes from the official model card.",
                    "Derived byte traffic is analytical and not an observed benchmark.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    baseline = HardwareProfile(
        name="RTX 4080 + i7-13700 + 32 GiB (assumption profile)",
        vram_gib=16.0,
        ram_gib=32.0,
        gpu_peak_bandwidth_gbps=716.8,
        gpu_efficiency=0.55,
        ram_effective_bandwidth_gbps=50.0,
        nvme_effective_bandwidth_gbps=5.5,
        pcie_effective_bandwidth_gbps=24.0,
        reserved_vram_gib=8.0,
        fixed_overhead_ms=1.5,
    )
    hardware_path = ROOT / "data/hardware/rtx4080_i7_13700_32gb_assumed.json"
    hardware_path.write_text(
        json.dumps(
            {
                **baseline.to_dict(spec),
                "measurement_status": "assumed_not_measured",
                "warning": (
                    "Replace effective bandwidths with host measurements before treating "
                    "throughput estimates as hardware-specific."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    throughput_rows: list[dict[str, object]] = []
    for gpu_efficiency in (0.35, 0.45, 0.55, 0.65, 0.75):
        for ram_bw in (35.0, 50.0, 65.0):
            for nvme_bw in (3.5, 5.5, 7.0):
                for overhead_ms in (0.0, 1.0, 2.0, 3.0):
                    profile = HardwareProfile(
                        name="synthetic-sweep",
                        vram_gib=16.0,
                        ram_gib=32.0,
                        gpu_peak_bandwidth_gbps=716.8,
                        gpu_efficiency=gpu_efficiency,
                        ram_effective_bandwidth_gbps=ram_bw,
                        nvme_effective_bandwidth_gbps=nvme_bw,
                        pcie_effective_bandwidth_gbps=24.0,
                        reserved_vram_gib=8.0,
                        fixed_overhead_ms=overhead_ms,
                    )
                    estimate = optimize_placement(spec, profile, step=0.01)
                    throughput_rows.append(
                        {
                            "gpu_efficiency": gpu_efficiency,
                            "gpu_effective_gbps": round(profile.gpu_effective_bandwidth_gbps, 4),
                            "ram_effective_gbps": ram_bw,
                            "nvme_effective_gbps": nvme_bw,
                            "fixed_overhead_ms": overhead_ms,
                            "best_gpu_fraction": round(estimate.placement.gpu, 4),
                            "best_ram_fraction": round(estimate.placement.ram, 4),
                            "best_nvme_fraction": round(estimate.placement.nvme, 4),
                            "gpu_time_ms": round(estimate.gpu_time_ms, 6),
                            "ram_time_ms": round(estimate.ram_time_ms, 6),
                            "nvme_time_ms": round(estimate.nvme_time_ms, 6),
                            "total_time_ms": round(estimate.total_time_ms, 6),
                            "estimated_tokens_per_second": round(
                                estimate.tokens_per_second, 6
                            ),
                            "data_class": "synthetic_analytical",
                        }
                    )
    write_csv(ROOT / "data/synthetic/throughput_sweep.csv", throughput_rows)

    zipf_rows: list[dict[str, object]] = []
    for exponent in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8):
        for cached in (4, 8, 12, 16, 20, 24, 32, 48, 64):
            hit = zipf_probability_mass(spec.experts_per_layer, cached, exponent)
            zipf_rows.append(
                {
                    "experts_per_layer": spec.experts_per_layer,
                    "cached_experts_per_layer": cached,
                    "zipf_exponent": exponent,
                    "per_selection_hit_probability": round(hit, 9),
                    "all_top4_hit_independence_approx": round(
                        all_active_experts_hit_approx(
                            hit, spec.active_experts_per_token
                        ),
                        9,
                    ),
                    "data_class": "synthetic_distribution",
                }
            )
    write_csv(ROOT / "data/synthetic/zipf_cache_sweep.csv", zipf_rows)

    prefetch_rows: list[dict[str, object]] = []
    tier_latencies = {
        "ram_local_read": 0.24,
        "ram_to_gpu_pcie": 0.55,
        "nvme_to_ram_optimistic": 2.21,
        "nvme_to_ram_tail": 4.50,
    }
    for target_tps in (10, 20, 30, 40, 60, 90):
        budget = layer_budget_ms(target_tps, spec.layers)
        for tier, latency_ms in tier_latencies.items():
            prefetch_rows.append(
                {
                    "target_tokens_per_second": target_tps,
                    "layers": spec.layers,
                    "layer_budget_ms": round(budget, 9),
                    "tier": tier,
                    "assumed_load_latency_ms": latency_ms,
                    "minimum_prefetch_horizon_layers": prefetch_horizon_layers(
                        latency_ms, budget
                    ),
                    "data_class": "synthetic_latency",
                }
            )
    write_csv(ROOT / "data/synthetic/prefetch_deadline_sweep.csv", prefetch_rows)

    miss_rows: list[dict[str, object]] = []
    for clean_probability in (0.90, 0.95, 0.99, 0.999, 0.9999):
        layer_p = max_layer_miss_probability(spec.layers, clean_probability)
        miss_rows.append(
            {
                "layers": spec.layers,
                "target_probability_token_has_no_critical_miss": clean_probability,
                "maximum_independent_layer_critical_miss_probability": round(
                    layer_p, 12
                ),
                "minimum_layer_critical_miss_avoidance": round(1.0 - layer_p, 12),
                "union_bound_approximation": round(
                    (1.0 - clean_probability) / spec.layers, 12
                ),
                "data_class": "analytical_requirement",
            }
        )
    write_csv(ROOT / "data/synthetic/critical_miss_requirements.csv", miss_rows)

    capacity_rows: list[dict[str, object]] = []
    for reserved in (6.0, 7.0, 8.0, 9.0, 10.0, 11.0):
        profile = HardwareProfile(
            name="capacity-sweep",
            vram_gib=16.0,
            ram_gib=32.0,
            gpu_peak_bandwidth_gbps=716.8,
            gpu_efficiency=0.55,
            ram_effective_bandwidth_gbps=50.0,
            nvme_effective_bandwidth_gbps=5.5,
            pcie_effective_bandwidth_gbps=24.0,
            reserved_vram_gib=reserved,
        )
        slots = profile.expert_slots(spec)
        capacity_rows.append(
            {
                "vram_gib": profile.vram_gib,
                "reserved_vram_gib": reserved,
                "expert_cache_gib": round(profile.expert_cache_bytes / (1024**3), 6),
                "expert_size_mib": round(spec.expert_size_mib, 9),
                "expert_slots_total": slots,
                "expert_slots_per_layer_mean": round(slots / spec.layers, 6),
                "data_class": "analytical_capacity",
            }
        )
    write_csv(ROOT / "data/synthetic/vram_capacity_sweep.csv", capacity_rows)

    baseline_estimate = optimize_placement(spec, baseline, step=0.0025)
    report = f"""# RTX 4080 baseline analytical report

This report is generated from assumptions, not a measured GPT-OSS-120B run.

## Inputs

- GPU peak bandwidth: {baseline.gpu_peak_bandwidth_gbps:.1f} GB/s
- Assumed GPU effective efficiency: {baseline.gpu_efficiency:.2%}
- Effective GPU bandwidth: {baseline.gpu_effective_bandwidth_gbps:.2f} GB/s
- Assumed RAM effective bandwidth: {baseline.ram_effective_bandwidth_gbps:.2f} GB/s
- Assumed NVMe effective path: {baseline.nvme_effective_bandwidth_gbps:.2f} GB/s
- Fixed software overhead: {baseline.fixed_overhead_ms:.2f} ms/token
- VRAM reserved outside expert cache: {baseline.reserved_vram_gib:.2f} GiB
- Estimated expert slots: {baseline.expert_slots(spec)} total, {baseline.expert_slots(spec) / spec.layers:.2f} per layer on average

## Derived model traffic

- Expert size: {spec.expert_size_mib:.4f} MiB
- Active expert traffic: {spec.active_expert_bytes_per_token / 1e9:.4f} GB/token
- Estimated dense traffic: {spec.estimated_dense_bytes_per_token / 1e9:.4f} GB/token
- Estimated total active traffic: {spec.estimated_total_active_bytes_per_token / 1e9:.4f} GB/token

## Grid-optimized idealized placement

- GPU fraction: {baseline_estimate.placement.gpu:.2%}
- RAM fraction: {baseline_estimate.placement.ram:.2%}
- NVMe fraction: {baseline_estimate.placement.nvme:.2%}
- Critical path before fixed overhead: {baseline_estimate.critical_path_ms:.4f} ms/token
- Total modeled time: {baseline_estimate.total_time_ms:.4f} ms/token
- Modeled throughput: {baseline_estimate.tokens_per_second:.2f} tok/s

## Interpretation

The estimate is an optimistic systems model. It assumes GPU, CPU/RAM, and NVMe
expert paths overlap perfectly and represents dequantization/compute through
*effective bandwidth*. It does not include router prediction errors, cache churn,
CUDA launch tails, OS page-fault tails, thermal throttling, or implementation
limitations. Replace assumed rates with observed measurements before drawing a
hardware conclusion.
"""
    (ROOT / "reports/rtx4080_baseline.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
