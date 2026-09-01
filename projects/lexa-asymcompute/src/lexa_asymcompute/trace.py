from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

_ALLOWED_FIELDS = {"token", "layer", "experts", "phase", "domain", "source", "batch_size", "schema_version"}
_FORBIDDEN_FIELDS = {"prompt", "text", "generated_text", "token_id", "logits", "hidden_state", "embedding", "api_key"}


@dataclass(frozen=True)
class RouteEvent:
    token: int
    layer: int
    experts: tuple[int, ...]
    phase: str = "decode"


def load_trace(path: str | Path, *, strict: bool = True) -> list[RouteEvent]:
    events: list[RouteEvent] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        keys = set(raw)
        if keys & _FORBIDDEN_FIELDS:
            raise ValueError(f"line {line_number}: forbidden trace field")
        if strict and keys - _ALLOWED_FIELDS:
            raise ValueError(f"line {line_number}: unknown trace field(s): {sorted(keys - _ALLOWED_FIELDS)}")
        token, layer = raw.get("token"), raw.get("layer")
        experts = raw.get("experts")
        if type(token) is not int or token < 0 or type(layer) is not int or layer < 0:
            raise ValueError(f"line {line_number}: token and layer must be non-negative integers")
        if not isinstance(experts, list) or not experts or any(type(item) is not int or item < 0 for item in experts):
            raise ValueError(f"line {line_number}: experts must be non-negative integer IDs")
        if len(experts) != len(set(experts)):
            raise ValueError(f"line {line_number}: duplicate Expert ID")
        events.append(RouteEvent(token=token, layer=layer, experts=tuple(experts), phase=raw.get("phase", "decode")))
    if not events:
        raise ValueError("trace is empty")
    return events


def validate_geometry(events: Iterable[RouteEvent], *, layers: int, top_k: int, experts_per_layer: int) -> dict[str, object]:
    items = list(events)
    seen: set[tuple[int, int]] = set()
    errors: list[str] = []
    tokens = sorted({event.token for event in items})
    for event in items:
        key = (event.token, event.layer)
        if key in seen:
            errors.append(f"duplicate event token={event.token} layer={event.layer}")
        seen.add(key)
        if not 0 <= event.layer < layers:
            errors.append(f"layer out of range: {event.layer}")
        if len(event.experts) != top_k:
            errors.append(f"wrong top-k at token={event.token} layer={event.layer}")
        if any(expert >= experts_per_layer for expert in event.experts):
            errors.append(f"Expert out of range at token={event.token} layer={event.layer}")
    expected = len(tokens) * layers
    if len(seen) != expected:
        errors.append(f"incomplete token-layer grid: observed={len(seen)} expected={expected}")
    return {"valid": not errors, "errors": errors, "tokens": len(tokens), "events": len(items)}


def temporal_top_n_holdout(
    events: Iterable[RouteEvent], *, cache_per_layer: int, calibration_fraction: float = 0.70
) -> dict[str, object]:
    if cache_per_layer < 0 or not 0 < calibration_fraction < 1:
        raise ValueError("invalid cache size or calibration fraction")
    items = sorted(events, key=lambda item: (item.token, item.layer))
    tokens = sorted({event.token for event in items})
    split_index = max(1, min(len(tokens) - 1, int(len(tokens) * calibration_fraction)))
    calibration_tokens = set(tokens[:split_index])
    evaluation_tokens = set(tokens[split_index:])

    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for event in items:
        if event.token in calibration_tokens:
            counts[event.layer].update(event.experts)
    cache = {layer: {expert for expert, _ in counter.most_common(cache_per_layer)} for layer, counter in counts.items()}

    selected = hits = full_hits = rows = 0
    for event in items:
        if event.token not in evaluation_tokens:
            continue
        layer_cache = cache.get(event.layer, set())
        row_hits = sum(expert in layer_cache for expert in event.experts)
        selected += len(event.experts)
        hits += row_hits
        full_hits += row_hits == len(event.experts)
        rows += 1
    return {
        "classification": "chronological_holdout_router_cache_measurement",
        "calibration_tokens": len(calibration_tokens),
        "evaluation_tokens": len(evaluation_tokens),
        "cache_per_layer": cache_per_layer,
        "selection_hit_rate": hits / selected if selected else 0.0,
        "full_row_hit_rate": full_hits / rows if rows else 0.0,
        "evaluation_rows": rows,
    }
