from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

_ALLOWED_CLASSES = {
    "observed_measurement",
    "observed_synthetic_harness",
    "analytical_estimate",
    "analytical_proxy",
}


@dataclass(frozen=True)
class EvidenceRecord:
    classification: str
    environment: str
    measured: bool
    notes: str = ""

    def validate(self) -> None:
        if self.classification not in _ALLOWED_CLASSES:
            raise ValueError(f"unsupported evidence classification: {self.classification}")
        if self.measured and not self.classification.startswith("observed_"):
            raise ValueError("measured records must use an observed classification")
        if not self.measured and self.classification.startswith("observed_"):
            raise ValueError("observed classifications must be measured")


def validate_evidence_payload(payload: Mapping[str, Any]) -> EvidenceRecord:
    record = EvidenceRecord(
        classification=str(payload["classification"]),
        environment=str(payload["environment"]),
        measured=bool(payload["measured"]),
        notes=str(payload.get("notes", "")),
    )
    record.validate()
    return record
