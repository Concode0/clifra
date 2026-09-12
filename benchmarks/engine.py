"""Common synchronization, warmup, and timing engine."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass

from .model import BenchmarkAdapter, BenchmarkRequest, ConstructedBenchmark, PreparedBenchmark
from .provenance import capture_provenance

SCHEMA_VERSION = "4.0.0"
STAGE_SEMANTICS = {
    "construction_setup": "backend context, contracts, and deterministic inputs; no preparation",
    "planning_preparation": "cold backend preparation on freshly constructed state; construction excluded",
    "first_invocation": "first forward call on a fresh prepared operation; planning excluded",
    "steady_forward": "forward call reusing one warmed prepared operation and fixed inputs",
    "forward_backward": "forward, squared-sum loss, and backward; gradient reset excluded",
}


@dataclass(frozen=True)
class MeasurementConfig:
    samples: int = 5
    warmups: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 1:
            raise ValueError("samples must be a positive integer")
        if isinstance(self.warmups, bool) or not isinstance(self.warmups, int) or self.warmups < 0:
            raise ValueError("warmups must be a non-negative integer")


@dataclass
class _LifecycleObservation:
    """State produced by the lifecycle that is actually being measured."""

    constructed: ConstructedBenchmark | None = None
    prepared: PreparedBenchmark | None = None


def _percentile(samples: list[int], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summary(samples: list[int]) -> dict[str, object]:
    median = statistics.median(samples)
    deviations = [abs(sample - median) for sample in samples]
    p25, p75 = _percentile(samples, 0.25), _percentile(samples, 0.75)
    return {
        "median_ns": median,
        "min_ns": min(samples),
        "dispersion": {
            "mad_ns": statistics.median(deviations),
            "p25_ns": p25,
            "p75_ns": p75,
            "iqr_ns": p75 - p25,
        },
    }


def _time(call, synchronize_call):
    synchronize_call()
    start = time.perf_counter_ns()
    result = call()
    synchronize_call()
    return time.perf_counter_ns() - start, result


def _measure(
    request: BenchmarkRequest,
    adapter: BenchmarkAdapter,
    config: MeasurementConfig,
    observation: _LifecycleObservation,
):
    effective_warmups = 0 if request.mode == "first_invocation" else config.warmups

    def synchronize_call():
        adapter.synchronize(request)

    if request.mode == "construction_setup":
        for _ in range(effective_warmups):
            adapter.construct(request)
            synchronize_call()
        samples = []
        for _ in range(config.samples):
            elapsed, constructed = _time(lambda: adapter.construct(request), synchronize_call)
            observation.constructed = constructed
            samples.append(elapsed)
        return samples, effective_warmups
    if request.mode == "planning_preparation":
        for _ in range(effective_warmups):
            constructed = adapter.construct(request)
            adapter.prepare(request, constructed)
            synchronize_call()
        samples = []
        for _ in range(config.samples):
            constructed = adapter.construct(request)
            observation.constructed = constructed
            elapsed, prepared = _time(
                lambda: adapter.prepare(request, constructed),
                synchronize_call,
            )
            observation.prepared = prepared
            samples.append(elapsed)
        return samples, effective_warmups
    if request.mode == "first_invocation":
        samples = []
        for _ in range(config.samples):
            prepared = adapter.prepare(request, adapter.construct(request))
            observation.constructed = prepared.constructed
            observation.prepared = prepared
            elapsed, _ = _time(
                lambda: adapter.invoke(prepared, backward=False),
                synchronize_call,
            )
            samples.append(elapsed)
        return samples, 0

    backward = request.mode == "forward_backward"
    prepared = adapter.prepare(request, adapter.construct(request, requires_grad=backward))
    observation.constructed = prepared.constructed
    observation.prepared = prepared

    for _ in range(effective_warmups):
        adapter.reset(prepared, backward=backward)
        adapter.invoke(prepared, backward=backward)
        synchronize_call()
    samples = []
    for _ in range(config.samples):
        adapter.reset(prepared, backward=backward)
        elapsed, _ = _time(
            lambda: adapter.invoke(prepared, backward=backward),
            synchronize_call,
        )
        samples.append(elapsed)
    return samples, effective_warmups


def _apply_observation(row: dict[str, object], observation: _LifecycleObservation) -> None:
    constructed = observation.constructed
    if constructed is not None:
        row["contracts"] = {
            "inputs": list(constructed.input_metadata),
            "output": constructed.output_metadata,
        }
    if observation.prepared is not None:
        row["backend_metadata"] = observation.prepared.metadata


def _local_fingerprint(request, config):
    payload = {
        **request.to_dict(),
        "measurement": {"samples": config.samples, "warmups": config.warmups},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _base_row(request, adapter, request_fingerprint, provenance):
    case, placement, mode = request.case, request.placement, request.mode
    ordinary_shapes = [list(item.leading_shape) for item in case.inputs]
    if case.family == "action" and case.operation == "linear":
        ordinary_shapes.append(list(case.ordinary_inputs[0].leading_shape))
    return {
        "schema_version": SCHEMA_VERSION,
        "row_id": request.row_id,
        "request_fingerprint": request_fingerprint,
        "case_id": case.case_id,
        "family": case.family,
        "operation": case.operation,
        "signature": {"p": case.signature[0], "q": case.signature[1], "r": case.signature[2]},
        "ordinary_leading_input_shapes": ordinary_shapes,
        "semantic_parameters": {
            "action_grade": case.action_grade,
            "generator_structure": case.generator_structure,
            "value_scale": case.value_scale,
        },
        "dtype": placement.dtype,
        "device": placement.device,
        "seed": case.seed,
        "backend": adapter.name,
        "contracts": {"inputs": [], "output": None},
        "backend_metadata": None,
        "timing": {
            "mode": mode,
            "stage_semantics": STAGE_SEMANTICS[mode],
            "clock": "time.perf_counter_ns",
            "synchronization": adapter.synchronization_description(request),
            "warmup_count": 0,
            "samples_ns": [],
            "summary_ns": None,
        },
        "status": "failed",
        "error": None,
        "provenance": provenance,
    }


def measure_request(
    request: BenchmarkRequest,
    adapter: BenchmarkAdapter,
    config: MeasurementConfig | None = None,
    *,
    request_fingerprint: str | None = None,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Measure one backend request in the current process."""

    config = MeasurementConfig() if config is None else config
    fingerprint = request_fingerprint or _local_fingerprint(request, config)
    row = _base_row(
        request,
        adapter,
        fingerprint,
        capture_provenance(request.placement.device) if provenance is None else provenance,
    )
    issue = adapter.preflight(request)
    if issue is not None:
        row["status"] = issue.status
        row["error"] = {"type": issue.error_type, "message": issue.message}
        return row
    observation = _LifecycleObservation()
    try:
        samples, warmups = _measure(request, adapter, config, observation)
        _apply_observation(row, observation)
        row["timing"]["warmup_count"] = warmups
        row["timing"]["samples_ns"] = samples
        row["timing"]["summary_ns"] = _summary(samples)
        row["status"] = "ok"
        return row
    except Exception as error:
        _apply_observation(row, observation)
        row["status"] = "failed"
        row["error"] = {"type": type(error).__name__, "message": str(error)}
        return row
