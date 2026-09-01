"""Causal cache-policy simulations for privacy-safe MoE router traces.

The static full-trace top-N calculation is an oracle upper bound because it sees
future decisions. These policies update residency only after each event or fit a
frozen cache on a chronological prefix.
"""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .trace import RouterEvent


@dataclass(frozen=True, slots=True)
class CachePolicyStats:
    policy: str
    capacity_per_layer: int
    events: int
    expert_selections: int
    selection_hits: int
    all_selected_hit_events: int
    miss_events: int
    expert_loads: int
    evictions: int
    evaluated_streams: int
    warmup_events: int = 0

    @property
    def per_selection_hit_rate(self) -> float:
        return self.selection_hits / self.expert_selections if self.expert_selections else 0.0

    @property
    def all_selected_hit_rate(self) -> float:
        return self.all_selected_hit_events / self.events if self.events else 0.0

    @property
    def miss_event_rate(self) -> float:
        return self.miss_events / self.events if self.events else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "capacity_per_layer": self.capacity_per_layer,
            "events": self.events,
            "expert_selections": self.expert_selections,
            "selection_hits": self.selection_hits,
            "per_selection_hit_rate": self.per_selection_hit_rate,
            "all_selected_hit_events": self.all_selected_hit_events,
            "all_selected_hit_rate": self.all_selected_hit_rate,
            "miss_events": self.miss_events,
            "miss_event_rate": self.miss_event_rate,
            "expert_loads": self.expert_loads,
            "evictions": self.evictions,
            "evaluated_streams": self.evaluated_streams,
            "warmup_events": self.warmup_events,
        }


def _stream(event: RouterEvent) -> str:
    return event.domain or "single-trace"


def _ordered(events: Iterable[RouterEvent]) -> tuple[RouterEvent, ...]:
    return tuple(sorted(events, key=lambda event: (_stream(event), event.token, event.layer)))


def _result(
    *,
    policy: str,
    capacity: int,
    events: int,
    selections: int,
    hits: int,
    all_hits: int,
    loads: int,
    evictions: int,
    streams: set[str],
    warmup_events: int = 0,
) -> CachePolicyStats:
    return CachePolicyStats(
        policy=policy,
        capacity_per_layer=capacity,
        events=events,
        expert_selections=selections,
        selection_hits=hits,
        all_selected_hit_events=all_hits,
        miss_events=events - all_hits,
        expert_loads=loads,
        evictions=evictions,
        evaluated_streams=len(streams),
        warmup_events=warmup_events,
    )


def simulate_lru(events: Iterable[RouterEvent], *, capacity_per_layer: int) -> CachePolicyStats:
    """Simulate one causal LRU cache per domain stream and layer."""
    if capacity_per_layer < 0:
        raise ValueError("capacity_per_layer cannot be negative")
    caches: dict[tuple[str, int], OrderedDict[int, None]] = defaultdict(OrderedDict)
    n_events = selections = hits = all_hits = loads = evictions = 0
    streams: set[str] = set()
    for event in _ordered(events):
        stream = _stream(event)
        streams.add(stream)
        cache = caches[(stream, event.layer)]
        selected_hits = [expert in cache for expert in event.experts]
        n_events += 1
        selections += len(event.experts)
        hits += sum(selected_hits)
        all_hits += int(all(selected_hits))
        # Refresh all pre-event hits before inserting misses. Otherwise a miss
        # earlier in the same Top-K set can evict a later pre-event hit.
        for expert, was_hit in zip(event.experts, selected_hits):
            if was_hit:
                cache.move_to_end(expert)
        for expert, was_hit in zip(event.experts, selected_hits):
            if was_hit:
                continue
            loads += 1
            if capacity_per_layer == 0:
                continue
            if len(cache) >= capacity_per_layer:
                cache.popitem(last=False)
                evictions += 1
            cache[expert] = None
    return _result(
        policy="online-lru",
        capacity=capacity_per_layer,
        events=n_events,
        selections=selections,
        hits=hits,
        all_hits=all_hits,
        loads=loads,
        evictions=evictions,
        streams=streams,
    )


def simulate_ema(
    events: Iterable[RouterEvent],
    *,
    capacity_per_layer: int,
    decay: float = 0.95,
) -> CachePolicyStats:
    """Simulate a causal exponentially-decayed frequency cache."""
    if capacity_per_layer < 0:
        raise ValueError("capacity_per_layer cannot be negative")
    if not 0.0 < decay <= 1.0:
        raise ValueError("decay must be in (0, 1]")
    scores: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    resident: dict[tuple[str, int], set[int]] = defaultdict(set)
    n_events = selections = hits = all_hits = loads = evictions = 0
    streams: set[str] = set()
    for event in _ordered(events):
        stream = _stream(event)
        streams.add(stream)
        key = (stream, event.layer)
        hot = resident[key]
        selected_hits = [expert in hot for expert in event.experts]
        n_events += 1
        selections += len(event.experts)
        hits += sum(selected_hits)
        all_hits += int(all(selected_hits))
        layer_scores = scores[key]
        for expert in tuple(layer_scores):
            value = layer_scores[expert] * decay
            if value < 1e-12:
                del layer_scores[expert]
            else:
                layer_scores[expert] = value
        for expert in event.experts:
            layer_scores[expert] = layer_scores.get(expert, 0.0) + 1.0
        next_hot = {
            expert
            for expert, _ in sorted(layer_scores.items(), key=lambda item: (-item[1], item[0]))[
                :capacity_per_layer
            ]
        }
        loads += len(next_hot - hot)
        evictions += len(hot - next_hot)
        resident[key] = next_hot
    return _result(
        policy=f"online-ema-{decay:g}",
        capacity=capacity_per_layer,
        events=n_events,
        selections=selections,
        hits=hits,
        all_hits=all_hits,
        loads=loads,
        evictions=evictions,
        streams=streams,
    )


def simulate_warmup_static(
    events: Iterable[RouterEvent],
    *,
    capacity_per_layer: int,
    warmup_fraction: float = 0.2,
    minimum_warmup_events: int = 8,
) -> CachePolicyStats:
    """Fit on an initial prefix per stream/layer, then score only the suffix."""
    if capacity_per_layer < 0:
        raise ValueError("capacity_per_layer cannot be negative")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if minimum_warmup_events < 0:
        raise ValueError("minimum_warmup_events cannot be negative")
    grouped: dict[tuple[str, int], list[RouterEvent]] = defaultdict(list)
    for event in _ordered(events):
        grouped[(_stream(event), event.layer)].append(event)
    n_events = selections = hits = all_hits = loads = 0
    warmup_total = 0
    streams: set[str] = set()
    for (stream, _), rows in grouped.items():
        streams.add(stream)
        if len(rows) <= 1:
            continue
        proposed = max(minimum_warmup_events, int(len(rows) * warmup_fraction))
        warmup_count = min(max(1, proposed), len(rows) - 1)
        warmup = rows[:warmup_count]
        evaluation = rows[warmup_count:]
        warmup_total += warmup_count
        counts = Counter(expert for event in warmup for expert in event.experts)
        hot = {expert for expert, _ in counts.most_common(capacity_per_layer)}
        loads += len(hot)
        for event in evaluation:
            selected_hits = [expert in hot for expert in event.experts]
            n_events += 1
            selections += len(event.experts)
            hits += sum(selected_hits)
            all_hits += int(all(selected_hits))
    return _result(
        policy=f"warmup-static-{warmup_fraction:g}",
        capacity=capacity_per_layer,
        events=n_events,
        selections=selections,
        hits=hits,
        all_hits=all_hits,
        loads=loads,
        evictions=0,
        streams=streams,
        warmup_events=warmup_total,
    )
