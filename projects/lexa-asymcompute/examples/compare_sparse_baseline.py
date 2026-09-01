from pathlib import Path

from lexa_asymcompute.metrics import latency_reduction, speedup
from lexa_asymcompute.moe import MoEHardwareSpec, MoEModelSpec, MoEPlacement, estimate_moe, optimize_moe

ROOT = Path(__file__).resolve().parents[1]
model = MoEModelSpec.from_json(ROOT / "data/workloads/sparse_moe_reference.json")
hardware = MoEHardwareSpec.from_json(ROOT / "data/hardware/moe_three_tier_assumed.json")
baseline = estimate_moe(model, hardware, MoEPlacement(0.0, 1.0, 0.0))
optimized = optimize_moe(model, hardware)
print({
    "baseline": baseline.to_dict(),
    "optimized": optimized.to_dict(),
    "speedup": speedup(baseline.total_ms, optimized.total_ms),
    "latency_reduction": latency_reduction(baseline.total_ms, optimized.total_ms),
})
