"""Benchmark case feasibility checks and target construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from clifra import make_algebra
from clifra.core import PlanningLimits
from clifra.core.foundation.basis import expand_output_grades
from clifra.core.planning.exp import select_bivector_exp_executor_family
from clifra.core.planning.policy import ProductExecutionPolicy

from .models import PreparedCase, ResourceConfig, SignatureSpec

UNLOCKED_LIMIT = 1 << 62
UNLOCKED_PLANNING_LIMITS = PlanningLimits(
    warn_lanes=UNLOCKED_LIMIT,
    max_lanes=UNLOCKED_LIMIT,
    warn_pairs=UNLOCKED_LIMIT,
    max_pairs=UNLOCKED_LIMIT,
)
UNLOCKED_PRODUCT_POLICY = ProductExecutionPolicy(full_table_max_lanes=UNLOCKED_LIMIT)
PRODUCT_KINDS = {
    "gp",
    "geometric_product",
    "wedge",
    "symmetric_product",
    "commutator_product",
    "anti_commutator_product",
    "left_contraction",
    "right_contraction",
}
UNARY_KINDS = {"identity", "reverse", "grade_involution", "clifford_conjugation", "grade_projection"}


class PreflightSkip(Exception):
    """A statically unsupported or unsafe benchmark case."""


@dataclass(frozen=True)
class FeasibleCase:
    """Resolved layouts and resource estimate for one case/signature."""

    selectors: dict[str, tuple[int, ...]]
    estimated_bytes: int


def make_benchmark_algebra(spec: SignatureSpec, *, device: str, dtype: torch.dtype):
    """Construct an algebra with clifra planning limits unlocked."""

    return make_algebra(
        spec.p,
        spec.q,
        spec.r,
        device=device,
        dtype=dtype,
        planning_limits=UNLOCKED_PLANNING_LIMITS,
        product_execution_policy=UNLOCKED_PRODUCT_POLICY,
    )


def preflight_case(
    case: dict[str, Any],
    spec: SignatureSpec,
    *,
    batch: int,
    channels: int,
    actions: int,
    pairs: int,
    dtype: torch.dtype,
    device: str,
    resources: ResourceConfig,
) -> FeasibleCase:
    """Resolve grade selectors and reject structurally invalid or unsafe cases."""

    kind = str(case["kind"])
    selectors: dict[str, tuple[int, ...]] = {}
    for key in ("left_grades", "right_grades", "input_grades", "parameter_grades"):
        if key in case:
            selectors[key] = resolve_grades(case[key], spec.n, name=key)

    if kind == "product":
        operation = str(case.get("operation", ""))
        if operation not in PRODUCT_KINDS:
            raise PreflightSkip(f"unsupported product operation {operation!r}")
        left = _required(selectors, "left_grades")
        right = _required(selectors, "right_grades")
        selectors["output_grades"] = resolve_output_grades(
            case.get("output_grades", "infer"), spec.n, left, right, operation
        )
    elif kind == "unary":
        operation = str(case.get("operation", ""))
        if operation not in UNARY_KINDS:
            raise PreflightSkip(f"unsupported unary operation {operation!r}")
        input_grades = _required(selectors, "input_grades")
        output = case.get("output_grades", "same")
        selectors["output_grades"] = (
            input_grades if output == "same" else resolve_grades(output, spec.n, name="output_grades")
        )
        if operation == "grade_projection" and not set(selectors["output_grades"]).issubset(input_grades):
            raise PreflightSkip("grade_projection output grades must be present in the input layout")
    elif kind in {"signature_norm", "pseudoscalar_product"}:
        input_grades = _required(selectors, "input_grades")
        if kind == "pseudoscalar_product":
            output = case.get("output_grades", [spec.n - grade for grade in input_grades])
            selectors["output_grades"] = resolve_grades(output, spec.n, name="output_grades")
    elif kind == "bivector_exp":
        if _required(selectors, "input_grades") != (2,):
            raise PreflightSkip("bivector_exp requires input_grades [2]")
        selectors["output_grades"] = resolve_grades(case.get("output_grades", "even"), spec.n, name="output_grades")
    elif kind == "sandwich_action":
        full = tuple(range(spec.n + 1))
        selectors.update(input_grades=full, output_grades=full, parameter_grades=full)
    elif kind in {"versor_action", "multi_versor_action"}:
        grade = int(case.get("grade", 2))
        if grade not in (1, 2):
            raise PreflightSkip("planned versor actions support grade 1 or 2")
        input_grades = _required(selectors, "input_grades")
        parameter_grades = selectors.get("parameter_grades", (grade,))
        if parameter_grades != (grade,):
            raise PreflightSkip(f"parameter_grades must be [{grade}] for this action")
        selectors["parameter_grades"] = parameter_grades
        selectors["output_grades"] = resolve_grades(
            case.get("output_grades", input_grades), spec.n, name="output_grades"
        )
    elif kind == "paired_bivector_action":
        input_grades = _required(selectors, "input_grades")
        parameter_grades = selectors.get("parameter_grades", (2,))
        if parameter_grades != (2,):
            raise PreflightSkip("paired_bivector_action requires parameter_grades [2]")
        selectors["parameter_grades"] = parameter_grades
        selectors["output_grades"] = resolve_grades(
            case.get("output_grades", input_grades), spec.n, name="output_grades"
        )
    else:
        raise PreflightSkip(f"unsupported case kind {kind!r}")

    estimated = estimate_case_bytes(
        kind,
        selectors,
        spec=spec,
        batch=batch,
        channels=channels,
        actions=actions,
        pairs=pairs,
        dtype=dtype,
        device=device,
        safety_factor=resources.safety_factor,
    )
    exp_family = exp_executor_family(kind, spec=spec, selectors=selectors, dtype=dtype, device=device)
    public_lanes = max((lane_count(spec.n, grades) for grades in selectors.values()), default=0)
    internal_lanes = implicit_internal_lanes(kind, spec=spec, selectors=selectors, exp_family=exp_family)
    if max(public_lanes, internal_lanes) > resources.max_layout_lanes:
        raise PreflightSkip(
            f"layout requires {max(public_lanes, internal_lanes)} lanes, "
            f"above benchmark cap {resources.max_layout_lanes}"
        )
    if estimated > resources.max_estimated_bytes:
        raise PreflightSkip(
            f"estimated allocation {estimated} exceeds benchmark cap {resources.max_estimated_bytes} bytes"
        )
    return FeasibleCase(selectors=selectors, estimated_bytes=estimated)


def resolve_grades(value: Any, n: int, *, name: str) -> tuple[int, ...]:
    """Resolve an explicit or symbolic grade selector."""

    if value == "full":
        grades = tuple(range(n + 1))
    elif value == "even":
        grades = tuple(range(0, n + 1, 2))
    elif isinstance(value, list):
        grades = tuple(sorted({int(grade) for grade in value}))
    elif isinstance(value, tuple):
        grades = tuple(sorted({int(grade) for grade in value}))
    else:
        raise PreflightSkip(f"{name} must be an array, 'full', or 'even'")
    if not grades:
        raise PreflightSkip(f"{name} must not be empty")
    invalid = [grade for grade in grades if grade < 0 or grade > n]
    if invalid:
        raise PreflightSkip(f"{name} contains grades unavailable for n={n}: {invalid}")
    return grades


def resolve_output_grades(
    value: Any, n: int, left: tuple[int, ...], right: tuple[int, ...], operation: str
) -> tuple[int, ...]:
    if value != "infer":
        return resolve_grades(value, n, name="output_grades")
    try:
        output = tuple(expand_output_grades(left, right, n, op=operation))
    except ValueError as exc:
        raise PreflightSkip(str(exc)) from exc
    if not output:
        raise PreflightSkip("operation has no structural output for the selected grades")
    return output


def estimate_case_bytes(
    kind: str,
    selectors: dict[str, tuple[int, ...]],
    *,
    spec: SignatureSpec,
    batch: int,
    channels: int,
    actions: int,
    pairs: int,
    dtype: torch.dtype,
    device: str,
    safety_factor: float,
) -> int:
    """Return a conservative pre-plan memory estimate."""

    n = spec.n
    element_size = torch.empty((), dtype=dtype).element_size()
    lanes = {name: lane_count(n, grades) for name, grades in selectors.items()}
    input_lanes = lanes.get("input_grades", 0)
    output_lanes = lanes.get("output_grades", input_lanes)
    parameter_lanes = lanes.get("parameter_grades", 0)
    tensor_elements = batch * max(input_lanes + output_lanes, 1)
    interaction_count = max(input_lanes, output_lanes, 1)

    if kind == "product":
        left_lanes = lanes["left_grades"]
        right_lanes = lanes["right_grades"]
        tensor_elements = batch * (left_lanes + right_lanes + output_lanes)
        interaction_count = left_lanes * right_lanes
    elif kind in {"versor_action", "multi_versor_action", "paired_bivector_action", "sandwich_action"}:
        tensor_elements = batch * channels * (input_lanes + output_lanes)
        if kind == "versor_action":
            tensor_elements += channels * parameter_lanes
        elif kind == "multi_versor_action":
            tensor_elements += actions * parameter_lanes + channels * actions
        elif kind == "paired_bivector_action":
            tensor_elements += 2 * pairs * parameter_lanes + channels
        else:
            tensor_elements += 2 * channels * input_lanes
        interaction_count = max(parameter_lanes * max(input_lanes, 1), input_lanes)
    elif kind == "bivector_exp":
        tensor_elements = batch * (input_lanes + output_lanes)
        interaction_count = max(input_lanes * output_lanes, n * n)

    exp_family = exp_executor_family(kind, spec=spec, selectors=selectors, dtype=dtype, device=device)
    if exp_family in {"left_matrix_exp", "cpu_matrix_exp"}:
        operator_lanes = 1 << (n - 1)
        tensor_elements += 32 * batch * operator_lanes * operator_lanes
    if kind == "paired_bivector_action":
        rotor_lanes = 1 << (n - 1)
        tensor_elements += 2 * pairs * rotor_lanes
        interaction_count = max(interaction_count, rotor_lanes * max(input_lanes, output_lanes, 1))

    tensor_bytes = tensor_elements * element_size
    plan_bytes = interaction_count * 32
    return int(math.ceil((tensor_bytes + plan_bytes) * float(safety_factor)))


def exp_executor_family(
    kind: str,
    *,
    spec: SignatureSpec,
    selectors: dict[str, tuple[int, ...]],
    dtype: torch.dtype,
    device: str,
) -> str | None:
    """Return the static exponential route used by a benchmark case kind."""

    if kind not in {"bivector_exp", "paired_bivector_action", "versor_action", "multi_versor_action"}:
        return None
    if kind in {"versor_action", "multi_versor_action"}:
        if selectors.get("input_grades") == (1,) and selectors.get("output_grades") == (1,):
            return "vector_matrix_exp"
    return select_bivector_exp_executor_family(spec, device, dtype=dtype)


def implicit_internal_lanes(
    kind: str,
    *,
    spec: SignatureSpec,
    selectors: dict[str, tuple[int, ...]],
    exp_family: str | None,
) -> int:
    """Return the largest implicit non-public layout used by a case."""

    if kind == "sandwich_action":
        return 1 << spec.n
    if kind == "paired_bivector_action":
        return 1 << (spec.n - 1)
    if kind in {"versor_action", "multi_versor_action"}:
        return spec.n if exp_family == "vector_matrix_exp" else 1 << (spec.n - 1)
    if kind == "bivector_exp":
        return 1 << (spec.n - 1) if exp_family in {"left_matrix_exp", "cpu_matrix_exp"} else 1
    return 0


def lane_count(n: int, grades: Iterable[int]) -> int:
    return sum(math.comb(n, int(grade)) for grade in grades)


def build_case(
    algebra,
    case: dict[str, Any],
    feasible: FeasibleCase,
    *,
    batch: int,
    channels: int,
    actions: int,
    pairs: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> PreparedCase:
    """Construct a planned callable and deterministic inputs."""

    kind = str(case["kind"])
    selectors = feasible.selectors
    layouts = {name: algebra.layout(grades) for name, grades in selectors.items()}
    scale = float(case.get("scale", 1.0))
    backward = bool(case.get("backward", False))
    metadata: dict[str, Any] = {
        "estimated_bytes": feasible.estimated_bytes,
        "backward_requested": backward,
    }
    for name, layout in layouts.items():
        prefix = name.removesuffix("_grades")
        metadata[f"{prefix}_grades"] = list(layout.grades)
        metadata[f"{prefix}_lanes"] = layout.dim

    if kind == "product":
        operation = str(case["operation"])
        left = layouts["left_grades"]
        right = layouts["right_grades"]
        output = layouts["output_grades"]
        module = algebra.plan_product(op=operation, left_layout=left, right_layout=right, output_layout=output)
        args = (
            random_tensor((batch, left.dim), device=device, dtype=dtype, seed=seed, scale=scale),
            random_tensor((batch, right.dim), device=device, dtype=dtype, seed=seed + 1, scale=scale),
        )
        metadata.update(executor_family=module.executor_family, pair_count=int(module.pair_count))
        return PreparedCase(str(case["id"]), kind, operation, module, args, left, output, metadata, backward)

    if kind == "unary":
        operation = str(case["operation"])
        input_layout = layouts["input_grades"]
        output = layouts["output_grades"]
        module = algebra.plan_unary(op=operation, input_layout=input_layout, output_layout=output)
        args = (random_tensor((batch, input_layout.dim), device=device, dtype=dtype, seed=seed, scale=scale),)
        metadata.update(executor_family=getattr(module, "executor_family", "unary"), pair_count=input_layout.dim)
        return PreparedCase(str(case["id"]), kind, operation, module, args, input_layout, output, metadata, backward)

    if kind == "signature_norm":
        input_layout = layouts["input_grades"]
        module = algebra.plan_signature_norm_squared(input_layout=input_layout)
        args = (random_tensor((batch, input_layout.dim), device=device, dtype=dtype, seed=seed, scale=scale),)
        metadata.update(
            executor_family=getattr(module, "executor_family", "metric"), output_lanes=1, pair_count=input_layout.dim
        )
        return PreparedCase(
            str(case["id"]), kind, "signature_norm_squared", module, args, input_layout, None, metadata, backward
        )

    if kind == "pseudoscalar_product":
        input_layout = layouts["input_grades"]
        output = layouts["output_grades"]
        module = algebra.plan_pseudoscalar_product(input_layout=input_layout, output_layout=output)
        args = (random_tensor((batch, input_layout.dim), device=device, dtype=dtype, seed=seed, scale=scale),)
        metadata.update(executor_family=getattr(module, "executor_family", "permutation"), pair_count=input_layout.dim)
        return PreparedCase(str(case["id"]), kind, kind, module, args, input_layout, output, metadata, backward)

    if kind == "bivector_exp":
        input_layout = layouts["input_grades"]
        output = layouts["output_grades"]
        module = algebra.plan_bivector_exp(input_layout=input_layout, output_layout=output)
        args = (random_tensor((batch, input_layout.dim), device=device, dtype=dtype, seed=seed, scale=scale),)
        metadata.update(
            executor_family=getattr(module, "executor_family", "bivector_exp"),
            exp_executor_family=getattr(module, "executor_family", "bivector_exp"),
            exp_operator_lanes=int(module.operator_layout.dim),
            pair_count=int(getattr(getattr(module, "left_product", None), "pair_count", 0) or 0),
        )
        return PreparedCase(str(case["id"]), kind, kind, module, args, input_layout, output, metadata, backward)

    if kind == "sandwich_action":
        layout = layouts["input_grades"]
        module = algebra.plan_sandwich_action(layout=layout)
        args = (
            random_tensor((channels, layout.dim), device=device, dtype=dtype, seed=seed, scale=scale),
            random_tensor((batch, channels, layout.dim), device=device, dtype=dtype, seed=seed + 1),
            random_tensor((channels, layout.dim), device=device, dtype=dtype, seed=seed + 2, scale=scale),
        )
        metadata.update(executor_family=getattr(module, "executor_family", "sandwich"), pair_count=layout.dim**2)
        return PreparedCase(str(case["id"]), kind, kind, module, args, layout, layout, metadata, backward)

    input_layout = layouts["input_grades"]
    output = layouts["output_grades"]
    parameter = layouts["parameter_grades"]
    values = random_tensor((batch, channels, input_layout.dim), device=device, dtype=dtype, seed=seed)

    if kind == "versor_action":
        grade = int(case.get("grade", 2))
        module = algebra.plan_versor_action(
            grade=grade,
            input_layout=input_layout,
            output_layout=output,
            parameter_layout=parameter,
        )
        args = (
            values,
            random_tensor((channels, parameter.dim), device=device, dtype=dtype, seed=seed + 1, scale=scale),
        )
    elif kind == "multi_versor_action":
        grade = int(case.get("grade", 2))
        module = algebra.plan_multi_versor_action(
            grade=grade,
            input_layout=input_layout,
            output_layout=output,
            parameter_layout=parameter,
        )
        args = (
            values,
            random_tensor((actions, parameter.dim), device=device, dtype=dtype, seed=seed + 1, scale=scale),
            random_tensor((channels, actions), device=device, dtype=dtype, seed=seed + 2),
        )
    elif kind == "paired_bivector_action":
        module = algebra.plan_paired_bivector_action(
            input_layout=input_layout,
            output_layout=output,
            parameter_layout=parameter,
        )
        args = (
            values,
            random_tensor((pairs, parameter.dim), device=device, dtype=dtype, seed=seed + 1, scale=scale),
            random_tensor((pairs, parameter.dim), device=device, dtype=dtype, seed=seed + 2, scale=scale),
            torch.arange(channels, device=device, dtype=torch.long) % pairs,
        )
    else:
        raise ValueError(f"unhandled benchmark case kind {kind!r}")
    metadata.update(
        executor_family=getattr(module, "executor_family", kind),
        pair_count=_action_pair_count(module),
        **_action_exp_metadata(module),
    )
    return PreparedCase(str(case["id"]), kind, kind, module, args, input_layout, output, metadata, backward)


def _action_exp_metadata(module) -> dict[str, Any]:
    executor = module.executor
    vector_matrix = getattr(executor, "vector_matrix", None)
    planned_exp = getattr(executor, "bivector_exp", None)
    if vector_matrix is not None and getattr(executor, "grade", 2) == 2:
        return {
            "action_execution_path": "vector_matrix",
            "exp_executor_family": "vector_matrix_exp",
            "matrix_exp_order": int(vector_matrix.n),
            "internal_rotor_lanes": 0,
            "internal_middle_lanes": 0,
        }
    metadata: dict[str, Any] = {
        "action_execution_path": "full_action" if getattr(executor, "use_full_action", False) else "rotor_product",
        "exp_executor_family": None,
        "matrix_exp_order": 0,
        "internal_rotor_lanes": int(getattr(getattr(executor, "rotor_layout", None), "dim", 0) or 0),
        "internal_middle_lanes": int(getattr(getattr(executor, "middle_layout", None), "dim", 0) or 0),
    }
    if planned_exp is not None:
        metadata["exp_executor_family"] = planned_exp.executor_family
        metadata["exp_operator_lanes"] = int(planned_exp.operator_layout.dim)
        if planned_exp.executor_family in {"left_matrix_exp", "cpu_matrix_exp"}:
            metadata["matrix_exp_order"] = int(planned_exp.operator_layout.dim)
    return metadata


def _action_pair_count(module) -> int:
    executor = module.executor
    return sum(
        int(getattr(product, "pair_count", 0) or 0)
        for product in (getattr(executor, "left_product", None), getattr(executor, "right_product", None))
    )


def random_tensor(
    shape: tuple[int, ...],
    *,
    device: str,
    dtype: torch.dtype,
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    values = torch.randn(*shape, dtype=torch.float64, generator=generator) * float(scale)
    return values.to(device=device, dtype=dtype)


def _required(selectors: dict[str, tuple[int, ...]], key: str) -> tuple[int, ...]:
    try:
        return selectors[key]
    except KeyError as exc:
        raise PreflightSkip(f"case requires {key}") from exc
