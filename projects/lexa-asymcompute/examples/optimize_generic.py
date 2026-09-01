from pathlib import Path

from lexa_asymcompute.models import HardwareSystem, WorkloadSpec
from lexa_asymcompute.optimizer import optimize

ROOT = Path(__file__).resolve().parents[1]
result = optimize(
    WorkloadSpec.from_json(ROOT / "data/workloads/edge_pipeline_example.json"),
    HardwareSystem.from_json(ROOT / "data/hardware/desktop_heterogeneous_example.json"),
    step=0.01,
)
print(result.to_dict())
