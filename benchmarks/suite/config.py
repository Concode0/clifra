"""Load and validate the benchmark JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clifra.core.foundation.device import FLOAT_DTYPES

from . import SCHEMA_VERSION
from .models import (
    BenchmarkConfig,
    OutputConfig,
    ProfilerConfig,
    ResourceConfig,
    SignatureSpec,
    SweepConfig,
    TimingConfig,
)

VALID_FAMILIES = {"euclidean", "mixed", "degenerate"}
VALID_MODES = {"eager", "inductor"}
VALID_PRESETS = {"full", "compact", "custom"}
VALID_KINDS = {
    "product",
    "unary",
    "signature_norm",
    "pseudoscalar_product",
    "bivector_exp",
    "sandwich_action",
    "versor_action",
    "multi_versor_action",
    "paired_bivector_action",
}


def load_config(path: Path) -> BenchmarkConfig:
    """Read and validate one benchmark configuration."""

    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("benchmark configuration must be a JSON object")
    version = _positive_int(raw.get("schema_version"), "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark schema_version {version}; expected {SCHEMA_VERSION}")

    sweeps = tuple(_load_sweep(value, index) for index, value in enumerate(_nonempty_list(raw, "sweeps")))
    sweep_ids = [sweep.sweep_id for sweep in sweeps]
    if len(sweep_ids) != len(set(sweep_ids)):
        raise ValueError("every sweep must have a unique id")

    timing_raw = _mapping(raw.get("timing"), "timing")
    timing = TimingConfig(
        warmup_calls=_nonnegative_int(timing_raw.get("warmup_calls"), "timing.warmup_calls"),
        samples=_positive_int(timing_raw.get("samples"), "timing.samples"),
        backward_samples=_positive_int(timing_raw.get("backward_samples"), "timing.backward_samples"),
    )
    resources_raw = _mapping(raw.get("resources"), "resources")
    resources = ResourceConfig(
        max_estimated_bytes=_positive_int(resources_raw.get("max_estimated_bytes"), "resources.max_estimated_bytes"),
        max_layout_lanes=_positive_int(resources_raw.get("max_layout_lanes"), "resources.max_layout_lanes"),
        safety_factor=_positive_float(resources_raw.get("safety_factor"), "resources.safety_factor"),
    )

    sweep_case_ids = {sweep.sweep_id: {str(case["id"]) for case in sweep.cases} for sweep in sweeps}
    cumulative = tuple(_mapping(value, "cumulative[]") for value in _list(raw, "cumulative"))
    for item in cumulative:
        sweep_id = str(item.get("sweep_id", ""))
        if sweep_id not in sweep_case_ids:
            raise ValueError(f"cumulative declaration references unknown sweep {sweep_id!r}")
        if item.get("case_id") not in sweep_case_ids[sweep_id]:
            raise ValueError(f"cumulative case {item.get('case_id')!r} is not declared in sweep {sweep_id!r}")
        sweep = next(sweep for sweep in sweeps if sweep.sweep_id == sweep_id)
        cumulative_dtypes = tuple(str(value) for value in _list(item, "dtypes")) or sweep.dtypes
        unknown_dtypes = sorted(set(cumulative_dtypes) - set(sweep.dtypes))
        if unknown_dtypes:
            raise ValueError(f"cumulative declaration references dtypes outside its sweep: {unknown_dtypes}")
        min_dimension = _positive_int(item.get("min_dimension", 1), "cumulative[].min_dimension")
        max_dimension = _positive_int(item.get("max_dimension", 63), "cumulative[].max_dimension")
        if min_dimension > max_dimension:
            raise ValueError("cumulative[].min_dimension must not exceed max_dimension")
        _positive_int(item.get("steps"), "cumulative[].steps")
        _positive_int(item.get("samples"), "cumulative[].samples")

    profiler_raw = _mapping(raw.get("profiler"), "profiler")
    profiler = ProfilerConfig(
        enabled=bool(profiler_raw.get("enabled", False)),
        case_ids=tuple(str(value) for value in _list(profiler_raw, "case_ids")),
        record_shapes=bool(profiler_raw.get("record_shapes", True)),
        profile_memory=bool(profiler_raw.get("profile_memory", True)),
    )
    all_case_ids = set().union(*sweep_case_ids.values())
    unknown_profile_cases = sorted(set(profiler.case_ids) - all_case_ids)
    if unknown_profile_cases:
        raise ValueError(f"profiler references unknown case ids: {unknown_profile_cases}")

    output_raw = _mapping(raw.get("output"), "output")
    baseline_value = output_raw.get("baseline")
    output = OutputConfig(
        root=Path(str(output_raw.get("root", "benchmarks/results"))),
        publish=bool(output_raw.get("publish", False)),
        docs_root=Path(str(output_raw.get("docs_root", "docs/benchmarks"))),
        baseline=None if baseline_value in (None, "") else Path(str(baseline_value)),
    )
    return BenchmarkConfig(
        schema_version=version,
        seed=int(raw.get("seed", 2026)),
        sweeps=sweeps,
        timing=timing,
        resources=resources,
        cumulative=cumulative,
        profiler=profiler,
        output=output,
        raw=raw,
    )


def expand_signatures(sweep: SweepConfig) -> tuple[SignatureSpec, ...]:
    """Expand dimension/family declarations and arbitrary explicit signatures."""

    result = list(sweep.signatures)
    for n in sweep.dimensions:
        if "euclidean" in sweep.signature_families:
            result.append(SignatureSpec(n, 0, 0))
        if "mixed" in sweep.signature_families and n >= 2:
            result.append(SignatureSpec(n - 1, 1, 0))
        if "degenerate" in sweep.signature_families and n >= 2:
            result.append(SignatureSpec(n - 1, 0, 1))
    unique: dict[tuple[int, int, int], SignatureSpec] = {}
    for spec in result:
        unique[(spec.p, spec.q, spec.r)] = spec
    return tuple(unique.values())


def _load_sweep(value: Any, index: int) -> SweepConfig:
    raw = _mapping(value, f"sweeps[{index}]")
    sweep_id = str(raw.get("id", ""))
    if not sweep_id:
        raise ValueError(f"sweeps[{index}].id must not be empty")
    preset = str(raw.get("layout_preset", "custom"))
    if preset not in VALID_PRESETS:
        raise ValueError(f"sweep {sweep_id!r} has unknown layout_preset {preset!r}")
    dimensions = _dimensions(raw.get("dimensions"), f"sweep {sweep_id!r}")
    families = tuple(str(item) for item in _list(raw, "signature_families"))
    unknown_families = sorted(set(families) - VALID_FAMILIES)
    if unknown_families:
        raise ValueError(f"sweep {sweep_id!r} has unknown signature families: {unknown_families}")
    signatures = tuple(_signature(item, sweep_id) for item in _list(raw, "signatures"))
    devices = tuple(str(item) for item in _nonempty_list(raw, "devices"))
    dtypes = tuple(str(item) for item in _nonempty_list(raw, "dtypes"))
    unknown_dtypes = sorted(set(dtypes) - set(FLOAT_DTYPES))
    if unknown_dtypes:
        raise ValueError(f"sweep {sweep_id!r} has unknown dtypes: {unknown_dtypes}")
    modes = tuple(str(item) for item in _nonempty_list(raw, "compile_modes"))
    unknown_modes = sorted(set(modes) - VALID_MODES)
    if unknown_modes:
        raise ValueError(f"sweep {sweep_id!r} has unknown compile modes: {unknown_modes}")
    batches = tuple(_positive_int(item, f"sweep {sweep_id!r} batch") for item in _nonempty_list(raw, "batch_sizes"))
    cases = tuple(_mapping(item, f"sweep {sweep_id!r} cases[]") for item in _nonempty_list(raw, "cases"))
    case_ids = [str(case.get("id", "")) for case in cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"sweep {sweep_id!r} case ids must be unique and non-empty")
    for case in cases:
        kind = str(case.get("kind", ""))
        if kind not in VALID_KINDS:
            raise ValueError(f"case {case['id']!r} has unknown kind {kind!r}")
        minimum_dimension = _positive_int(case.get("min_dimension", 1), f"case {case['id']!r} min_dimension")
        maximum_dimension = _positive_int(case.get("max_dimension", 63), f"case {case['id']!r} max_dimension")
        if minimum_dimension > maximum_dimension or maximum_dimension > 63:
            raise ValueError(f"case {case['id']!r} has an invalid dimension range")
        maximum_mixed_dimension = _positive_int(
            case.get("max_mixed_dimension", maximum_dimension),
            f"case {case['id']!r} max_mixed_dimension",
        )
        if maximum_mixed_dimension > maximum_dimension:
            raise ValueError(f"case {case['id']!r} max_mixed_dimension exceeds its maximum dimension")
        _validate_preset_case(preset, case, sweep_id)
    sweep = SweepConfig(
        sweep_id=sweep_id,
        layout_preset=preset,
        dimensions=dimensions,
        signature_families=families,
        signatures=signatures,
        devices=devices,
        dtypes=dtypes,
        batch_sizes=batches,
        compile_modes=modes,
        channels=_positive_int(raw.get("channels"), f"sweep {sweep_id!r} channels"),
        actions=_positive_int(raw.get("actions"), f"sweep {sweep_id!r} actions"),
        pairs=_positive_int(raw.get("pairs"), f"sweep {sweep_id!r} pairs"),
        cases=cases,
    )
    if not expand_signatures(sweep):
        raise ValueError(f"sweep {sweep_id!r} resolves to no signatures")
    return sweep


def _validate_preset_case(preset: str, case: dict[str, Any], sweep_id: str) -> None:
    if preset != "full":
        return
    kind = str(case["kind"])
    for key in ("left_grades", "right_grades", "input_grades", "output_grades"):
        if key in case and case[key] not in ("full", "same", "infer"):
            raise ValueError(f"full-layout sweep {sweep_id!r} case {case['id']!r} must use symbolic full layouts")
    if kind in {"versor_action", "multi_versor_action", "paired_bivector_action"}:
        if case.get("input_grades") != "full" or case.get("output_grades") != "full":
            raise ValueError(f"full-layout action {case['id']!r} must use full input and output layouts")


def case_applies(case: dict[str, Any], spec: SignatureSpec) -> bool:
    """Return whether a case is defined for the signature dimension."""

    if not int(case.get("min_dimension", 1)) <= spec.n <= int(case.get("max_dimension", 63)):
        return False
    if spec.p > 0 and spec.q > 0:
        return spec.n <= int(case.get("max_mixed_dimension", case.get("max_dimension", 63)))
    return True


def _dimensions(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, list):
        dimensions = tuple(_positive_int(item, f"{name} dimensions[]") for item in value)
    elif isinstance(value, dict):
        minimum = _positive_int(value.get("min"), f"{name} dimensions.min")
        maximum = _positive_int(value.get("max"), f"{name} dimensions.max")
        step = _positive_int(value.get("step", 1), f"{name} dimensions.step")
        if maximum < minimum:
            raise ValueError(f"{name} dimensions.max must be >= dimensions.min")
        dimensions = tuple(range(minimum, maximum + 1, step))
    else:
        raise ValueError(f"{name} dimensions must be an array or min/max object")
    if any(dimension > 63 for dimension in dimensions):
        raise ValueError(f"{name} dimensions must not exceed n=63")
    return dimensions


def _signature(value: Any, sweep_id: str) -> SignatureSpec:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"sweep {sweep_id!r} signatures entries must be [p, q, r]")
    p, q, r = (int(part) for part in value)
    if min(p, q, r) < 0 or not 1 <= p + q + r <= 63:
        raise ValueError(f"invalid signature {(p, q, r)}")
    return SignatureSpec(p, q, r)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _list(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    return value


def _nonempty_list(mapping: dict[str, Any], key: str) -> list[Any]:
    values = _list(mapping, key)
    if not values:
        raise ValueError(f"{key} must not be empty")
    return values


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result
