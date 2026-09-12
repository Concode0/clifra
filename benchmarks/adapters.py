"""Narrow family adapters over clifra's public plans and private route seam.

Normal rows use public ``AlgebraContext.plan_*`` methods. Forced rows replace
only the root family's policy decision; nested families continue to use the
unchanged ``DefaultPolicy``. Capability and resources are always assessed by
the registered provider before a forced route can execute.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

from clifra.core import AlgebraContext, TensorContract
from clifra.core._kernel.configuration import configured_algebra
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, PolicyEvaluation
from clifra.core._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS, ResourceRequirements
from clifra.core._kernel.providers import (
    BuiltinProvider,
    action_execution_request,
    exp_execution_request,
    product_execution_request,
)
from clifra.core.executors import Rejected

from .cases import BenchmarkCase, ExecutionPlacement, LayoutCase
from .model import (
    BenchmarkRequest,
    ConstructedBenchmark,
    MeasurementIssue,
    PreparedBenchmark,
)


@dataclass(frozen=True)
class _RequestedRootRoutePolicy:
    family: str
    route: str

    def evaluate(self, candidate):
        if candidate.family != self.family:
            return DEFAULT_PLANNING_POLICY.evaluate(candidate)
        if candidate.route == self.route:
            return PolicyEvaluation(0.0, "benchmark_requested_root_route")
        return PolicyEvaluation(None, "benchmark_other_root_route")


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _algebra(case: BenchmarkCase, placement: ExecutionPlacement):
    kwargs = {"device": placement.device, "dtype": torch_dtype(placement.dtype)}
    if placement.selection.mode == "default":
        return AlgebraContext(*case.signature, **kwargs)
    return configured_algebra(
        *case.signature,
        **kwargs,
        planning_policy=_RequestedRootRoutePolicy(case.family, placement.selection.route),
    )


def _contract(algebra, declaration: LayoutCase) -> TensorContract:
    layout = algebra.layout(declaration.grades)
    constructor = TensorContract.canonical if declaration.storage == "canonical" else TensorContract.compact
    return constructor(layout)


def declarations(algebra, case: BenchmarkCase):
    clifford = tuple(_contract(algebra, declaration) for declaration in case.inputs)
    inputs = (clifford[0], None) if case.family == "action" and case.operation == "linear" else clifford
    return inputs, _contract(algebra, case.output)


def _random_compact(declaration, contract, generator, *, scale=1.0):
    return scale * torch.randn(
        *declaration.leading_shape,
        contract.layout.dim,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )


def _structured_clifford(case, index, declaration, contract, generator):
    scale = case.value_scale if case.family == "bivector_exp" or index > 0 else 1.0
    values = _random_compact(declaration, contract, generator, scale=scale)
    structure = case.generator_structure
    if structure in {"simple_plane", "commuting_planes", "null_plane"} and (case.family == "bivector_exp" or index > 0):
        values = torch.zeros_like(values)
        positions = {blade: position for position, blade in enumerate(contract.layout.basis_indices)}
        if structure == "null_plane":
            blades = ((1 << 0) | (1 << (sum(case.signature) - 1)),)
        elif structure == "simple_plane":
            blades = (3,)
        else:
            blades = tuple(3 << (2 * plane) for plane in range(sum(case.signature) // 2))
        for ordinal, blade in enumerate(blades):
            if blade in positions:
                values[..., positions[blade]] = scale / (ordinal + 1)
    elif structure == "reflection_normal" and index > 0:
        values = torch.zeros_like(values)
        values[..., 0] = 1.0
    return contract.layout.full(values) if contract.uses_canonical_storage else values


def construct_case(
    case: BenchmarkCase,
    placement: ExecutionPlacement = ExecutionPlacement(),
    *,
    requires_grad: bool = False,
):
    """Construct declarations and deterministic inputs without planning."""

    algebra = _algebra(case, placement)
    inputs, output = declarations(algebra, case)
    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    values = []
    clifford_index = 0
    for contract in inputs:
        if contract is None:
            declaration = case.ordinary_inputs[0]
            value = case.value_scale * torch.randn(
                *declaration.leading_shape,
                *declaration.trailing_shape,
                generator=generator,
                dtype=torch.float64,
            )
            if case.generator_structure == "near_identity_matrix":
                value = value + torch.eye(sum(case.signature), dtype=torch.float64)
        else:
            declaration = case.inputs[clifford_index]
            value = _structured_clifford(case, clifford_index, declaration, contract, generator)
            clifford_index += 1
        values.append(
            value.to(device=placement.device, dtype=torch_dtype(placement.dtype)).detach().requires_grad_(requires_grad)
        )
    return algebra, inputs, output, tuple(values)


def plan_case(algebra, inputs, output, case: BenchmarkCase):
    """Plan through the normal public family surface."""

    if case.family == "product":
        return algebra.plan_product(op=case.operation, left=inputs[0], right=inputs[1], output=output)
    if case.family == "bivector_exp":
        return algebra.plan_bivector_exp(input=inputs[0], output=output)
    if case.operation == "linear":
        return algebra.plan_linear_action(input=inputs[0], output=output)
    return algebra.plan_versor_action(
        grade=case.action_grade,
        input=inputs[0],
        parameter=inputs[1],
        output=output,
    )


def _execution_request(algebra, inputs, output, case):
    dtype, device = algebra.dtype, algebra.device
    if case.family == "product":
        declaration = ProductRequest.compact(
            algebra.spec,
            op=case.operation,
            left_layout=inputs[0].layout,
            right_layout=inputs[1].layout,
            output_layout=output.layout,
            dtype=dtype,
            device=device,
        )
        return product_execution_request(declaration)
    if case.family == "bivector_exp":
        return exp_execution_request(algebra.spec, device, dtype, output.layout, planner=algebra._planner)
    parameter = None if case.operation == "linear" else inputs[1].layout
    return action_execution_request(
        algebra,
        case.operation,
        grade=case.action_grade,
        input_layout=inputs[0].layout,
        output_layout=output.layout,
        parameter_layout=parameter,
    )


@lru_cache(maxsize=None)
def route_assessments(case: BenchmarkCase, placement: ExecutionPlacement) -> tuple[dict[str, object], ...]:
    """Assess every registered built-in root route for this exact workload."""

    default = ExecutionPlacement(placement.dtype, placement.device)
    algebra = _algebra(case, default)
    inputs, output = declarations(algebra, case)
    request = _execution_request(algebra, inputs, output, case)
    result = []
    for provider in algebra.registry.providers:
        if not isinstance(provider, BuiltinProvider) or provider.identity[0] != case.family:
            continue
        assessment = provider.assess(request)
        if isinstance(assessment, Rejected):
            result.append(
                {
                    "route": provider.identity[1],
                    "eligible": False,
                    "ineligible_reason": assessment.reason,
                    "structural_facts": {},
                    "resource_requirements": {},
                }
            )
            continue
        resources = ResourceRequirements(assessment.lanes, assessment.pairs)
        reason = resources.rejection_reason(DEFAULT_RESOURCE_LIMITS)
        facts = assessment.preparation.facts
        result.append(
            {
                "route": provider.identity[1],
                "eligible": reason is None,
                "ineligible_reason": reason,
                "structural_facts": {} if facts is None else asdict(facts),
                "resource_requirements": asdict(resources),
            }
        )
    return tuple(result)


def forced_route_ineligibility(case: BenchmarkCase, placement: ExecutionPlacement) -> str | None:
    if placement.selection.mode != "forced_repository_private":
        return None
    assessment = next(
        (item for item in route_assessments(case, placement) if item["route"] == placement.selection.route),
        None,
    )
    if assessment is None:
        return f"unknown built-in route {placement.selection.route!r}"
    return assessment["ineligible_reason"]


def _kernel_route(kernel, case):
    route = getattr(kernel, "route", None)
    if route is None and case.family == "action" and case.operation == "linear":
        return "graded_linear"
    return route


def _child(kernel, path, family, operation=None):
    if kernel is None:
        return None
    kernel = getattr(kernel, "_kernel", kernel)
    route = getattr(kernel, "route", None)
    if route is None:
        route = {"unary": "grade_map", "metric": "diagonal"}.get(family)
    return {
        "path": path,
        "family": family,
        "operation": operation or getattr(kernel, "op", "unknown"),
        "selected_route": route,
        "executor_class": type(kernel).__name__,
    }


def _child_selections(kernel, case):
    children = []
    if case.family == "bivector_exp":
        for name, family, operation in (
            ("left_product", "product", "geometric_product"),
            ("bivector_wedge", "product", "wedge"),
            ("grade4_square", "product", "geometric_product"),
            ("bivector_grade4_product", "product", "geometric_product"),
            ("square", "product", "symmetric_product"),
        ):
            item = _child(getattr(kernel, name, None), name, family, operation)
            if item is not None:
                children.append(item)
        polynomial = getattr(kernel, "polynomial", None)
        if polynomial is not None:
            for index, product in enumerate(polynomial.products):
                children.append(_child(product, f"polynomial.products[{index}]", "product"))
    elif case.family == "action" and case.operation == "versor":
        for name, family in (
            ("bivector_exp", "bivector_exp"),
            ("rotor_reverse", "unary"),
            ("parameter_reverse", "unary"),
            ("parameter_signature_norm_squared", "metric"),
            ("input_involution", "unary"),
            ("left_product", "product"),
            ("right_product", "product"),
        ):
            item = _child(getattr(kernel, name, None), name, family)
            if item is not None:
                children.append(item)
    return children


def planning_metadata(operation, case: BenchmarkCase, placement: ExecutionPlacement) -> dict[str, object]:
    """Record root assessment facts plus observable composite child selections."""

    kernel = operation._kernel
    route = _kernel_route(kernel, case)
    assessment = next((item for item in route_assessments(case, placement) if item["route"] == route), None)
    return {
        "selection_mode": placement.selection.mode,
        "requested_route": placement.selection.route,
        "selected_family": case.family,
        "selected_route": route,
        "executor_class": type(kernel).__name__,
        "structural_facts": {} if assessment is None else assessment["structural_facts"],
        "resource_requirements": {} if assessment is None else assessment["resource_requirements"],
        "child_selections": _child_selections(kernel, case),
        "planner_cache_state": "fresh_for_cold_stages; reused_for_steady_stages",
    }


def describe_contract(contract) -> dict[str, object] | None:
    if contract is None:
        return None
    return {
        "layout_kind": type(contract.layout).__name__,
        "grades": list(contract.grades),
        "layout_dim": contract.layout.dim,
        "basis_indices": list(contract.layout.basis_indices),
        "storage": contract.storage.value,
        "lane_dim": contract.lane_dim,
    }


def _synchronize_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class ClifraAdapter:
    """Translate benchmark requests into fixed clifra executions."""

    name = "clifra"

    def preflight(self, request: BenchmarkRequest) -> MeasurementIssue | None:
        placement = request.placement
        unavailable = {
            "mps": not torch.backends.mps.is_available(),
            "cuda": not torch.cuda.is_available(),
        }.get(placement.device, False)
        if unavailable:
            return MeasurementIssue(
                "unsupported",
                "UnsupportedDevice",
                f"{placement.device.upper()} is not available in this process",
            )
        if placement.device in {"mps", "cuda"}:
            try:
                torch.empty(
                    (),
                    device=placement.device,
                    dtype=torch_dtype(placement.dtype),
                )
                _synchronize_device(placement.device)
            except (RuntimeError, TypeError) as error:
                return MeasurementIssue("unsupported", type(error).__name__, str(error))
        ineligible_reason = forced_route_ineligibility(request.case, placement)
        if ineligible_reason is not None:
            return MeasurementIssue("unsupported", "IneligibleRoute", ineligible_reason)
        return None

    def construct(self, request: BenchmarkRequest, *, requires_grad: bool = False) -> ConstructedBenchmark:
        algebra, inputs, output, values = construct_case(
            request.case,
            request.placement,
            requires_grad=requires_grad,
        )
        return ConstructedBenchmark(
            state=(algebra, inputs, output),
            arguments=values,
            input_metadata=tuple(describe_contract(contract) for contract in inputs),
            output_metadata=describe_contract(output),
        )

    def prepare(self, request: BenchmarkRequest, constructed: ConstructedBenchmark) -> PreparedBenchmark:
        algebra, inputs, output = constructed.state
        operation = plan_case(algebra, inputs, output, request.case)
        return PreparedBenchmark(
            operation=operation,
            constructed=constructed,
            metadata=planning_metadata(operation, request.case, request.placement),
        )

    def invoke(self, prepared: PreparedBenchmark, *, backward: bool) -> None:
        result = prepared.operation(*prepared.constructed.arguments)
        if backward:
            result.square().sum().backward()

    def reset(self, prepared: PreparedBenchmark, *, backward: bool) -> None:
        if backward:
            for value in prepared.constructed.arguments:
                value.grad = None

    def synchronize(self, request: BenchmarkRequest) -> None:
        _synchronize_device(request.placement.device)

    def synchronization_description(self, request: BenchmarkRequest) -> str:
        device = request.placement.device
        return f"torch.{device}.synchronize before and after" if device in {"mps", "cuda"} else "CPU synchronous"


CLIFRA_ADAPTER = ClifraAdapter()


def configure_clifra_runtime(cpu_threads: int, interop_threads: int) -> None:
    """Apply campaign thread controls to the current clifra/PyTorch process."""

    if cpu_threads < 1 or interop_threads < 1:
        raise ValueError("thread counts must be positive")
    os.environ.update(
        {
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
        }
    )
    torch.set_num_threads(cpu_threads)
    if torch.get_num_interop_threads() != interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "PyTorch interop threads cannot be changed after parallel work has started in this process"
            ) from error
