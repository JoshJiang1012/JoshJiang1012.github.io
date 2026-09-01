from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

_BYTES_PER_GB = 1_000_000_000.0
_FLOPS_PER_TFLOP = 1_000_000_000_000.0


@dataclass(frozen=True)
class DeviceSpec:
    """One compute tier or execution device.

    Values are deliberately *effective* modelling inputs when efficiencies are
    calibrated. Peak vendor specifications may be supplied with efficiency
    factors below one, but must never be labelled as measured throughput.
    """

    name: str
    compute_tflops: float
    memory_bandwidth_gbps: float
    memory_gib: float
    compute_efficiency: float = 1.0
    bandwidth_efficiency: float = 1.0
    fixed_latency_ms: float = 0.0
    power_watts: float = 0.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("device name must be non-empty")
        if self.compute_tflops <= 0 or self.memory_bandwidth_gbps <= 0:
            raise ValueError(f"device {self.name!r} needs positive compute and bandwidth")
        if self.memory_gib <= 0:
            raise ValueError(f"device {self.name!r} needs positive memory capacity")
        if not 0 < self.compute_efficiency <= 1:
            raise ValueError("compute_efficiency must be in (0, 1]")
        if not 0 < self.bandwidth_efficiency <= 1:
            raise ValueError("bandwidth_efficiency must be in (0, 1]")
        if self.fixed_latency_ms < 0 or self.power_watts < 0:
            raise ValueError("latency and power cannot be negative")

    @property
    def effective_compute_flops(self) -> float:
        return self.compute_tflops * self.compute_efficiency * _FLOPS_PER_TFLOP

    @property
    def effective_bandwidth_bytes_s(self) -> float:
        return self.memory_bandwidth_gbps * self.bandwidth_efficiency * _BYTES_PER_GB


@dataclass(frozen=True)
class LinkSpec:
    source: str
    target: str
    bandwidth_gbps: float
    latency_ms: float = 0.0
    energy_pj_per_byte: float = 0.0

    def validate(self) -> None:
        if not self.source or not self.target or self.source == self.target:
            raise ValueError("link must connect two different named devices")
        if self.bandwidth_gbps <= 0:
            raise ValueError("link bandwidth must be positive")
        if self.latency_ms < 0 or self.energy_pj_per_byte < 0:
            raise ValueError("link latency and energy cannot be negative")

    def transfer_ms(self, byte_count: float) -> float:
        if byte_count < 0:
            raise ValueError("transfer byte count cannot be negative")
        return self.latency_ms + 1000.0 * byte_count / (self.bandwidth_gbps * _BYTES_PER_GB)


@dataclass(frozen=True)
class HardwareSystem:
    name: str
    devices: tuple[DeviceSpec, ...]
    links: tuple[LinkSpec, ...] = ()

    def validate(self) -> None:
        if not self.name or not self.devices:
            raise ValueError("hardware system requires a name and at least one device")
        names = [device.name for device in self.devices]
        if len(names) != len(set(names)):
            raise ValueError("device names must be unique")
        for device in self.devices:
            device.validate()
        known = set(names)
        for link in self.links:
            link.validate()
            if link.source not in known or link.target not in known:
                raise ValueError("link endpoint is not a declared device")

    @classmethod
    def from_json(cls, path: str | Path) -> "HardwareSystem":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            devices=tuple(DeviceSpec(**item) for item in raw["devices"]),
            links=tuple(LinkSpec(**item) for item in raw.get("links", [])),
        )

    def device_map(self) -> dict[str, DeviceSpec]:
        return {device.name: device for device in self.devices}

    def link(self, source: str, target: str) -> LinkSpec | None:
        for link in self.links:
            if link.source == source and link.target == target:
                return link
        for link in self.links:
            if link.source == target and link.target == source:
                return LinkSpec(
                    source=source,
                    target=target,
                    bandwidth_gbps=link.bandwidth_gbps,
                    latency_ms=link.latency_ms,
                    energy_pj_per_byte=link.energy_pj_per_byte,
                )
        return None


@dataclass(frozen=True)
class WorkloadSpec:
    """A divisible workload with optional mandatory and serial components."""

    name: str
    parallel_flops: float
    parallel_bytes: float
    parallel_resident_bytes: float = 0.0
    serial_flops: float = 0.0
    serial_bytes: float = 0.0
    serial_device: str | None = None
    output_bytes: float = 0.0
    aggregate_device: str | None = None
    sync_overhead_ms: float = 0.0
    eligible_devices: tuple[str, ...] = ()
    mandatory_flops_by_device: Mapping[str, float] = field(default_factory=dict)
    mandatory_bytes_by_device: Mapping[str, float] = field(default_factory=dict)
    mandatory_resident_bytes_by_device: Mapping[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        numeric = (
            self.parallel_flops,
            self.parallel_bytes,
            self.parallel_resident_bytes,
            self.serial_flops,
            self.serial_bytes,
            self.output_bytes,
            self.sync_overhead_ms,
        )
        if not self.name or any(value < 0 for value in numeric):
            raise ValueError("workload name must be non-empty and quantities non-negative")
        for mapping in (
            self.mandatory_flops_by_device,
            self.mandatory_bytes_by_device,
            self.mandatory_resident_bytes_by_device,
        ):
            if any(value < 0 for value in mapping.values()):
                raise ValueError("mandatory work values cannot be negative")

    @classmethod
    def from_json(cls, path: str | Path) -> "WorkloadSpec":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["eligible_devices"] = tuple(raw.get("eligible_devices", ()))
        return cls(**raw)


@dataclass(frozen=True)
class FractionalPlacement:
    shares: Mapping[str, float]

    def validate(self, eligible: set[str]) -> None:
        if not self.shares:
            raise ValueError("placement cannot be empty")
        if set(self.shares) - eligible:
            raise ValueError("placement contains an ineligible or unknown device")
        if any(value < 0 or value > 1 for value in self.shares.values()):
            raise ValueError("placement shares must be in [0, 1]")
        if abs(sum(self.shares.values()) - 1.0) > 1e-9:
            raise ValueError("placement shares must sum to one")


@dataclass(frozen=True)
class DeviceEstimate:
    device: str
    share: float
    compute_ms: float
    memory_ms: float
    service_ms: float
    transfer_ms: float
    path_ms: float
    energy_j: float
    resident_gib: float


@dataclass(frozen=True)
class Estimate:
    workload: str
    hardware: str
    placement: FractionalPlacement
    devices: tuple[DeviceEstimate, ...]
    serial_ms: float
    parallel_critical_ms: float
    sync_overhead_ms: float
    total_ms: float
    units_per_second: float
    energy_j: float
    classification: str = "analytical_estimate_not_observed_benchmark"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["placement"] = dict(self.placement.shares)
        result["devices"] = [asdict(item) for item in self.devices]
        return result


def _service_components(device: DeviceSpec, flops: float, byte_count: float) -> tuple[float, float, float]:
    compute_ms = 1000.0 * flops / device.effective_compute_flops
    memory_ms = 1000.0 * byte_count / device.effective_bandwidth_bytes_s
    return compute_ms, memory_ms, max(compute_ms, memory_ms) + device.fixed_latency_ms


def estimate(
    workload: WorkloadSpec,
    hardware: HardwareSystem,
    placement: FractionalPlacement,
) -> Estimate:
    workload.validate()
    hardware.validate()
    devices = hardware.device_map()
    eligible = set(workload.eligible_devices or devices.keys())
    if not eligible <= set(devices):
        raise ValueError("workload lists a device not present in hardware")
    placement.validate(eligible)

    aggregate = workload.aggregate_device
    if aggregate is not None and aggregate not in devices:
        raise ValueError("aggregate_device is not present in hardware")

    estimates: list[DeviceEstimate] = []
    for name in sorted(placement.shares):
        share = placement.shares[name]
        device = devices[name]
        flops = share * workload.parallel_flops + workload.mandatory_flops_by_device.get(name, 0.0)
        byte_count = share * workload.parallel_bytes + workload.mandatory_bytes_by_device.get(name, 0.0)
        resident = (
            share * workload.parallel_resident_bytes
            + workload.mandatory_resident_bytes_by_device.get(name, 0.0)
        )
        resident_gib = resident / (1024.0**3)
        if resident_gib > device.memory_gib + 1e-9:
            raise ValueError(
                f"placement exceeds {name} capacity: {resident_gib:.3f} GiB > {device.memory_gib:.3f} GiB"
            )
        compute_ms, memory_ms, service_ms = _service_components(device, flops, byte_count)

        transfer_ms = 0.0
        transfer_energy_j = 0.0
        if aggregate and name != aggregate and workload.output_bytes > 0 and share > 0:
            link = hardware.link(name, aggregate)
            if link is None:
                raise ValueError(f"missing link from {name} to aggregate device {aggregate}")
            transfer_bytes = share * workload.output_bytes
            transfer_ms = link.transfer_ms(transfer_bytes)
            transfer_energy_j = link.energy_pj_per_byte * transfer_bytes * 1e-12

        path_ms = service_ms + transfer_ms
        energy_j = device.power_watts * service_ms / 1000.0 + transfer_energy_j
        estimates.append(
            DeviceEstimate(
                device=name,
                share=share,
                compute_ms=compute_ms,
                memory_ms=memory_ms,
                service_ms=service_ms,
                transfer_ms=transfer_ms,
                path_ms=path_ms,
                energy_j=energy_j,
                resident_gib=resident_gib,
            )
        )

    serial_ms = 0.0
    serial_energy = 0.0
    if workload.serial_flops or workload.serial_bytes:
        serial_name = workload.serial_device or aggregate
        if serial_name is None or serial_name not in devices:
            raise ValueError("serial work requires a valid serial_device or aggregate_device")
        _, _, serial_ms = _service_components(
            devices[serial_name], workload.serial_flops, workload.serial_bytes
        )
        serial_energy = devices[serial_name].power_watts * serial_ms / 1000.0

    parallel_critical = max((item.path_ms for item in estimates), default=0.0)
    total_ms = serial_ms + parallel_critical + workload.sync_overhead_ms
    if total_ms <= 0:
        raise ValueError("workload has zero estimated duration")
    return Estimate(
        workload=workload.name,
        hardware=hardware.name,
        placement=placement,
        devices=tuple(estimates),
        serial_ms=serial_ms,
        parallel_critical_ms=parallel_critical,
        sync_overhead_ms=workload.sync_overhead_ms,
        total_ms=total_ms,
        units_per_second=1000.0 / total_ms,
        energy_j=serial_energy + sum(item.energy_j for item in estimates),
    )
