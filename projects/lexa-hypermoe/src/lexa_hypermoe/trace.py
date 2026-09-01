from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


_ALLOWED_FIELDS = {
    "schema_version",
    "token",
    "layer",
    "experts",
    "domain",
    "phase",
    "source",
}


@dataclass(frozen=True)
class TraceRecord:
    token: int
    layer: int
    experts: tuple[int, ...]


def read_trace(path: str | Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        item = json.loads(raw)
        unknown = set(item) - _ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"line {line_number}: forbidden fields: {sorted(unknown)}")
        records.append(
            TraceRecord(
                token=int(item["token"]),
                layer=int(item["layer"]),
                experts=tuple(int(value) for value in item["experts"]),
            )
        )
    return records


def validate_trace(
    records: Iterable[TraceRecord],
    *,
    layers: int,
    experts_per_layer: int,
    active_experts: int,
) -> list[TraceRecord]:
    checked = list(records)
    if not checked:
        raise ValueError("trace is empty")
    seen: set[tuple[int, int]] = set()
    tokens: set[int] = set()
    for record in checked:
        key = (record.token, record.layer)
        if key in seen:
            raise ValueError(f"duplicate token/layer record: {key}")
        seen.add(key)
        tokens.add(record.token)
        if not 0 <= record.layer < layers:
            raise ValueError(f"invalid layer: {record.layer}")
        if len(record.experts) != active_experts:
            raise ValueError("unexpected active Expert count")
        if len(set(record.experts)) != len(record.experts):
            raise ValueError("Expert IDs must be unique per record")
        if any(not 0 <= expert < experts_per_layer for expert in record.experts):
            raise ValueError("Expert ID out of range")
    expected = len(tokens) * layers
    if len(checked) != expected:
        raise ValueError(f"incomplete trace geometry: {len(checked)} != {expected}")
    return checked


def holdout_topn(
    records: Iterable[TraceRecord],
    *,
    cached_experts_per_layer: int,
    calibration_fraction: float = 0.70,
) -> dict[str, float | int | str]:
    records = list(records)
    token_ids = sorted({record.token for record in records})
    if len(token_ids) < 2:
        raise ValueError("at least two tokens are required for holdout evaluation")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be within (0, 1)")
    split = min(len(token_ids) - 1, max(1, int(len(token_ids) * calibration_fraction)))
    calibration_tokens = set(token_ids[:split])
    holdout_tokens = set(token_ids[split:])

    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        if record.token in calibration_tokens:
            counts[record.layer].update(record.experts)
    cache = {
        layer: {expert for expert, _ in counter.most_common(cached_experts_per_layer)}
        for layer, counter in counts.items()
    }

    selections = hits = records_seen = all_hit_records = 0
    for record in records:
        if record.token not in holdout_tokens:
            continue
        layer_cache = cache.get(record.layer, set())
        record_hits = sum(expert in layer_cache for expert in record.experts)
        selections += len(record.experts)
        hits += record_hits
        records_seen += 1
        all_hit_records += int(record_hits == len(record.experts))

    return {
        "classification": "observed_router_holdout_not_throughput_benchmark",
        "calibration_tokens": len(calibration_tokens),
        "holdout_tokens": len(holdout_tokens),
        "holdout_records": records_seen,
        "selection_hit_rate": hits / selections if selections else 0.0,
        "all_experts_hit_rate": all_hit_records / records_seen if records_seen else 0.0,
    }
