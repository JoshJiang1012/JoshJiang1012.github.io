"""Router-trace ingestion and cache-hit analysis.

Trace rows are expected to be JSON Lines objects containing at least:

    {"token": 0, "layer": 0, "experts": [3, 8, 41, 90]}

Optional fields such as domain, latency, cache tier, and timestamp are retained
by callers but are not needed for the frequency analysis implemented here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RouterEvent:
    token: int
    layer: int
    experts: tuple[int, ...]
    domain: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "RouterEvent":
        token = int(value["token"])
        layer = int(value["layer"])
        raw_experts = value["experts"]
        if not isinstance(raw_experts, Sequence) or isinstance(raw_experts, (str, bytes)):
            raise ValueError("experts must be a sequence of integer IDs")
        experts = tuple(int(item) for item in raw_experts)
        if token < 0 or layer < 0 or not experts or any(item < 0 for item in experts):
            raise ValueError("token, layer, and expert IDs must be nonnegative")
        domain_raw = value.get("domain")
        domain = str(domain_raw) if domain_raw is not None else None
        return cls(token=token, layer=layer, experts=experts, domain=domain)


@dataclass(frozen=True, slots=True)
class LayerCacheStats:
    layer: int
    events: int
    expert_selections: int
    cached_experts: int
    per_selection_hit_rate: float
    all_selected_hit_rate: float
    hottest_experts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "events": self.events,
            "expert_selections": self.expert_selections,
            "cached_experts": self.cached_experts,
            "per_selection_hit_rate": self.per_selection_hit_rate,
            "all_selected_hit_rate": self.all_selected_hit_rate,
            "hottest_experts": list(self.hottest_experts),
        }


def iter_jsonl(path: str | Path) -> Iterator[RouterEvent]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise ValueError("row root must be an object")
                yield RouterEvent.from_mapping(raw)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid router trace at line {line_number}: {exc}") from exc


def layer_cache_stats(
    events: Iterable[RouterEvent],
    *,
    cached_experts_per_layer: int,
) -> tuple[LayerCacheStats, ...]:
    if cached_experts_per_layer < 0:
        raise ValueError("cached_experts_per_layer cannot be negative")
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    rows: dict[int, list[RouterEvent]] = defaultdict(list)
    for event in events:
        rows[event.layer].append(event)
        counts[event.layer].update(event.experts)

    result: list[LayerCacheStats] = []
    for layer in sorted(rows):
        hottest = tuple(
            expert for expert, _ in counts[layer].most_common(cached_experts_per_layer)
        )
        hot_set = set(hottest)
        layer_events = rows[layer]
        selected = sum(len(event.experts) for event in layer_events)
        hits = sum(
            sum(expert in hot_set for expert in event.experts)
            for event in layer_events
        )
        all_hit = sum(
            all(expert in hot_set for expert in event.experts)
            for event in layer_events
        )
        result.append(
            LayerCacheStats(
                layer=layer,
                events=len(layer_events),
                expert_selections=selected,
                cached_experts=cached_experts_per_layer,
                per_selection_hit_rate=(hits / selected if selected else 0.0),
                all_selected_hit_rate=(all_hit / len(layer_events) if layer_events else 0.0),
                hottest_experts=hottest,
            )
        )
    return tuple(result)


def aggregate_cache_stats(stats: Iterable[LayerCacheStats]) -> dict[str, float]:
    rows = tuple(stats)
    total_events = sum(item.events for item in rows)
    total_selections = sum(item.expert_selections for item in rows)
    weighted_selection_hits = sum(
        item.per_selection_hit_rate * item.expert_selections for item in rows
    )
    weighted_all_hits = sum(item.all_selected_hit_rate * item.events for item in rows)
    return {
        "layers": float(len(rows)),
        "events": float(total_events),
        "expert_selections": float(total_selections),
        "per_selection_hit_rate": (
            weighted_selection_hits / total_selections if total_selections else 0.0
        ),
        "all_selected_hit_rate": (
            weighted_all_hits / total_events if total_events else 0.0
        ),
    }
