"""Privacy-preserving router-trace ingestion and cache-locality analysis.

The v2 trace contract intentionally stores only numeric routing metadata:
sequence index, layer index, selected expert IDs, phase, batch size, a coarse
caller-supplied domain label, and the producing tensor name. Prompt text, token
IDs, logits, probabilities, embeddings, and hidden states are forbidden.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


TRACE_SCHEMA_VERSION = "2.0"
ALLOWED_PHASES = frozenset({"prefill", "decode"})
ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "token",
        "layer",
        "experts",
        "domain",
        "phase",
        "batch_size",
        "source",
        "timestamp_ns",
        "gpu_hits",
        "ram_hits",
        "nvme_misses",
        "layer_latency_us",
    }
)
FORBIDDEN_FIELDS = frozenset(
    {
        "prompt",
        "text",
        "content",
        "token_id",
        "token_ids",
        "token_piece",
        "logit",
        "logits",
        "probability",
        "probabilities",
        "embedding",
        "embeddings",
        "hidden_state",
        "hidden_states",
        "residual",
        "residuals",
        "api_key",
        "secret",
    }
)


def _sequence_of_ints(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field} must be a sequence of integer IDs")
    converted = tuple(int(item) for item in value)
    if not converted:
        raise ValueError(f"{field} must not be empty")
    if any(item < 0 for item in converted):
        raise ValueError(f"{field} IDs must be nonnegative")
    if len(converted) != len(set(converted)):
        raise ValueError(f"{field} IDs must be unique within one routing event")
    return converted


def validate_privacy_fields(value: Mapping[str, object], *, strict: bool = True) -> None:
    """Reject data fields that could disclose prompts or model internals.

    `strict=True` additionally rejects unknown top-level keys. This keeps the
    default ingestion path fail-closed when a future collector starts emitting
    more data than the published schema permits.
    """

    keys = {str(key) for key in value}
    forbidden = sorted(keys & FORBIDDEN_FIELDS)
    if forbidden:
        raise ValueError("privacy-forbidden fields present: " + ", ".join(forbidden))
    if strict:
        unknown = sorted(keys - ALLOWED_FIELDS)
        if unknown:
            raise ValueError("unknown trace fields present: " + ", ".join(unknown))


@dataclass(frozen=True, slots=True)
class RouterEvent:
    token: int
    layer: int
    experts: tuple[int, ...]
    domain: str | None = None
    phase: str | None = None
    batch_size: int | None = None
    source: str | None = None
    schema_version: str = TRACE_SCHEMA_VERSION

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        strict_privacy: bool = True,
    ) -> "RouterEvent":
        validate_privacy_fields(value, strict=strict_privacy)
        token = int(value["token"])
        layer = int(value["layer"])
        experts = _sequence_of_ints(value["experts"], field="experts")
        if token < 0 or layer < 0:
            raise ValueError("token and layer must be nonnegative")

        domain_raw = value.get("domain")
        domain = str(domain_raw) if domain_raw is not None else None
        if domain is not None and len(domain) > 128:
            raise ValueError("domain must be at most 128 characters")

        phase_raw = value.get("phase")
        phase = str(phase_raw) if phase_raw is not None else None
        if phase is not None and phase not in ALLOWED_PHASES:
            raise ValueError("phase must be prefill or decode")

        batch_raw = value.get("batch_size")
        batch_size = int(batch_raw) if batch_raw is not None else None
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive")

        source_raw = value.get("source")
        source = str(source_raw) if source_raw is not None else None
        if source is not None and len(source) > 128:
            raise ValueError("source must be at most 128 characters")

        schema_version = str(value.get("schema_version", "1.0"))
        if len(schema_version) > 16:
            raise ValueError("schema_version is too long")

        return cls(
            token=token,
            layer=layer,
            experts=experts,
            domain=domain,
            phase=phase,
            batch_size=batch_size,
            source=source,
            schema_version=schema_version,
        )


@dataclass(frozen=True, slots=True)
class LayerCacheStats:
    layer: int
    events: int
    expert_selections: int
    cached_experts: int
    unique_experts_seen: int
    per_selection_hit_rate: float
    all_selected_hit_rate: float
    hottest_experts: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "events": self.events,
            "expert_selections": self.expert_selections,
            "cached_experts": self.cached_experts,
            "unique_experts_seen": self.unique_experts_seen,
            "per_selection_hit_rate": self.per_selection_hit_rate,
            "all_selected_hit_rate": self.all_selected_hit_rate,
            "hottest_experts": list(self.hottest_experts),
        }


@dataclass(frozen=True, slots=True)
class LayerTemporalStats:
    layer: int
    events: int
    consecutive_pairs: int
    mean_overlap_fraction: float
    exact_repeat_rate: float

    def to_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "events": self.events,
            "consecutive_pairs": self.consecutive_pairs,
            "mean_overlap_fraction": self.mean_overlap_fraction,
            "exact_repeat_rate": self.exact_repeat_rate,
        }


@dataclass(frozen=True, slots=True)
class TraceAudit:
    path: str
    sha256: str
    bytes: int
    events: int
    tokens: int
    layers: int
    expert_selections: int
    max_expert_id: int | None
    phases: tuple[str, ...]
    domains: tuple[str, ...]
    schema_versions: tuple[str, ...]
    privacy_safe: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "events": self.events,
            "tokens": self.tokens,
            "layers": self.layers,
            "expert_selections": self.expert_selections,
            "max_expert_id": self.max_expert_id,
            "phases": list(self.phases),
            "domains": list(self.domains),
            "schema_versions": list(self.schema_versions),
            "privacy_safe": self.privacy_safe,
            "data_class": "observed_router_metadata_if_collected_from_runtime",
        }


def iter_jsonl(
    path: str | Path,
    *,
    phase: str | None = None,
    strict_privacy: bool = True,
) -> Iterator[RouterEvent]:
    if phase is not None and phase not in ALLOWED_PHASES:
        raise ValueError("phase must be prefill or decode")
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
                event = RouterEvent.from_mapping(raw, strict_privacy=strict_privacy)
                if phase is None or event.phase == phase:
                    yield event
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid router trace at line {line_number}: {exc}") from exc


def trace_audit(path: str | Path, *, strict_privacy: bool = True) -> TraceAudit:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    events = tuple(iter_jsonl(source, strict_privacy=strict_privacy))
    tokens = {event.token for event in events}
    layers = {event.layer for event in events}
    phases = tuple(sorted({event.phase for event in events if event.phase is not None}))
    domains = tuple(sorted({event.domain for event in events if event.domain is not None}))
    versions = tuple(sorted({event.schema_version for event in events}))
    expert_ids = [expert for event in events for expert in event.experts]
    return TraceAudit(
        path=str(source),
        sha256=digest.hexdigest(),
        bytes=source.stat().st_size,
        events=len(events),
        tokens=len(tokens),
        layers=len(layers),
        expert_selections=len(expert_ids),
        max_expert_id=max(expert_ids) if expert_ids else None,
        phases=phases,
        domains=domains,
        schema_versions=versions,
    )


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
                unique_experts_seen=len(counts[layer]),
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


def cache_sweep(
    events: Iterable[RouterEvent],
    *,
    cache_sizes: Iterable[int],
) -> tuple[dict[str, object], ...]:
    rows = tuple(events)
    result: list[dict[str, object]] = []
    for cache_size in sorted(set(int(item) for item in cache_sizes)):
        if cache_size < 0:
            raise ValueError("cache sizes cannot be negative")
        stats = layer_cache_stats(rows, cached_experts_per_layer=cache_size)
        result.append(
            {
                "cached_experts_per_layer": cache_size,
                "aggregate": aggregate_cache_stats(stats),
            }
        )
    return tuple(result)


def layer_temporal_stats(events: Iterable[RouterEvent]) -> tuple[LayerTemporalStats, ...]:
    rows: dict[int, list[RouterEvent]] = defaultdict(list)
    for event in events:
        rows[event.layer].append(event)

    result: list[LayerTemporalStats] = []
    for layer in sorted(rows):
        ordered = sorted(rows[layer], key=lambda item: item.token)
        overlap_total = 0.0
        exact_repeats = 0
        pairs = 0
        for previous, current in zip(ordered, ordered[1:]):
            if current.token == previous.token:
                continue
            previous_set = set(previous.experts)
            current_set = set(current.experts)
            denominator = max(len(previous_set), len(current_set), 1)
            overlap_total += len(previous_set & current_set) / denominator
            exact_repeats += previous_set == current_set
            pairs += 1
        result.append(
            LayerTemporalStats(
                layer=layer,
                events=len(ordered),
                consecutive_pairs=pairs,
                mean_overlap_fraction=(overlap_total / pairs if pairs else 0.0),
                exact_repeat_rate=(exact_repeats / pairs if pairs else 0.0),
            )
        )
    return tuple(result)
