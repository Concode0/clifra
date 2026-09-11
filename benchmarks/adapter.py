"""The only benchmark module coupled to private clifra planning interfaces."""

import hashlib
import math
from dataclasses import asdict, dataclass, fields

import torch

from clifra.core import PlannedOperation, TensorContract
from clifra.core._kernel.planning.layouts import ProductRequest
from clifra.core._kernel.planning.policy import DEFAULT_PLANNING_POLICY, PolicyEvaluation
from clifra.core._kernel.planning.unary import UnaryRequest
from clifra.core._kernel.providers import (
    UnaryExecutionRequest,
    action_execution_request,
    exp_execution_request,
    product_execution_request,
)
from clifra.core._kernel.routing import Selection
from clifra.core.executors import ExecutorRequest

from .timing import timed


class Unsupported(Exception):
    """Concrete capability rejection, distinct from a failed correctness check."""


def temporary_workarounds(case, metadata, args):
    if case.family not in {"action", "bivector_exp"}:
        return []
    parameter = args[1] if case.family == "action" else args[0]
    if math.prod(parameter.shape[:-1]) != 1:
        return []

    def uses_matrix_exp(node):
        matrix_action = (
            node.get("family") == "action" and node.get("route") == "vector_matrix" and case.inputs[1] == (2,)
        )
        return (
            matrix_action
            or (node.get("family"), node.get("route")) == ("bivector_exp", "left_matrix_exp")
            or any(uses_matrix_exp(child) for child in node.get("children", ()))
        )

    return ["pytorch_singleton_matrix_exp_batch_duplication"] if uses_matrix_exp(metadata) else []


@dataclass(frozen=True)
class ForceRoot:
    route: str

    def evaluate(self, candidate):
        return PolicyEvaluation(0.0 if candidate.route == self.route else None, "forced_root_route")


def contracts(algebra, case, storage):
    constructor = TensorContract.canonical if storage == "canonical" else TensorContract.compact
    return tuple(constructor(algebra.layout(g)) for g in case.inputs), constructor(algebra.layout(case.output))


def make_inputs(algebra, case, storage, max_bytes):
    inputs, _ = contracts(algebra, case, storage)
    output_width = (
        algebra.n + 1
        if case.operation == "lane_grade_energy"
        else (algebra.dim if storage == "canonical" else algebra.layout(case.output).dim)
    )
    leading = (
        (torch.broadcast_shapes(*(s[:-1] for s in case.leading)) + (case.leading[0][-1], case.leading[1][-1]))
        if case.pairwise
        else torch.broadcast_shapes(*case.leading)
    )
    # A caller-allocation guard, not an alternative route feasibility model.
    elements = sum(math.prod(s) * c.lane_dim for s, c in zip(case.leading, inputs))
    elements += math.prod(leading) * output_width
    if elements * torch.empty((), dtype=algebra.dtype).element_size() * 4 > max_bytes:
        raise Unsupported("harness_input_output_budget_exceeded")
    seed = int.from_bytes(hashlib.sha256(case.id.encode()).digest()[:8], "little") % (2**63)
    generator = torch.Generator().manual_seed(seed)
    values = []
    for i, (shape, contract) in enumerate(zip(case.leading, inputs)):
        x = torch.randn(*shape, contract.layout.dim, generator=generator, dtype=torch.float64)
        is_generator = case.family == "bivector_exp" or case.family == "action" and i == 1
        if is_generator:
            basis = contract.layout.basis_indices
            if case.structure in {"simple", "planes", "null"}:
                x.zero_()
                axes = (
                    [(0, 1)]
                    if case.structure == "simple"
                    else (
                        [(0, algebra.n - 1)]
                        if case.structure == "null"
                        else [(j, j + 1) for j in range(0, algebra.n - 1, 2)]
                    )
                )
                for j, (a, b) in enumerate(axes):
                    x[..., basis.index((1 << a) | (1 << b))] = j + 1
            x = x / x.abs().sum(-1, keepdim=True).clamp_min(1e-30) * case.magnitude
            if case.structure == "mixed":
                x = x * torch.linspace(0.1, 1.0, x.shape[0]).unsqueeze(-1)
        if case.family == "geometry" and i == 1:
            x[..., 0] += 3
        x = x.to(device=algebra.device, dtype=algebra.dtype)
        values.append(contract.layout.full(x) if storage == "canonical" else x)
    return tuple(values), seed


def public_operation(algebra, case, storage):
    ins, out = contracts(algebra, case, storage)
    if case.family == "product":
        return algebra.plan_product(op=case.operation, left=ins[0], right=ins[1], output=out, pairwise=case.pairwise)
    if case.family == "bivector_exp":
        return algebra.plan_bivector_exp(input=ins[0], output=out)
    if case.family == "action":
        return algebra.plan_versor_action(grade=case.inputs[1][0], input=ins[0], parameter=ins[1], output=out)
    if case.family == "unary":
        return algebra.plan_unary(op=case.operation, input=ins[0], output=out)
    if case.family in {"metric", "permutation"}:
        return getattr(algebra, "plan_" + case.operation)(input=ins[0], output=out)
    if case.family == "geometry":
        return algebra.plan_reflect(input=ins[0], normal=ins[1], output=out)
    if case.operation == "lane_grade_energy":
        return lambda x: algebra.lane_grade_energy(x, input=ins[0])
    return lambda x, y: algebra.conjugate_scalar_form(x, y, left=ins[0], right=ins[1])


def request_for(algebra, case):
    ins, out = contracts(algebra, case, "compact")
    dtype, device = algebra.dtype, algebra.device
    if case.family == "product":
        return product_execution_request(ProductRequest(algebra.spec, case.operation, *ins, out, dtype, device))
    if case.family == "bivector_exp":
        return exp_execution_request(algebra.spec, device, dtype, out.layout, planner=algebra._planner)
    if case.family == "action":
        return action_execution_request(
            algebra,
            "versor",
            grade=case.inputs[1][0],
            input_layout=ins[0].layout,
            parameter_layout=ins[1].layout,
            output_layout=out.layout,
        )
    args = (case.family, case.operation, ins, out, dtype, device)
    if case.family == "unary":
        return UnaryExecutionRequest(*args, UnaryRequest(algebra.spec, case.operation, ins[0], out, dtype, device))
    return ExecutorRequest(*args)


def selection_metadata(selection):
    children = []
    preparation = selection.assessment.preparation
    for field in fields(preparation):
        value = getattr(preparation, field.name)
        for child in value if isinstance(value, tuple) else (value,):
            if isinstance(child, Selection):
                children.append({"role": field.name, **selection_metadata(child)})
    return {
        "family": selection.family,
        "route": selection.route,
        "facts": asdict(selection.facts) if selection.facts is not None else None,
        "requirements": {"lanes": selection.assessment.lanes, "pairs": selection.assessment.pairs},
        "children": children,
    }


def prepare(algebra, case, route, storage):
    stages = {}
    if case.family in {"forms", "geometry"}:
        if route != "default":
            raise Unsupported("operation_has_no_comparable_root_route")
        op, stages["public_preparation_ms"] = timed(lambda: public_operation(algebra, case, storage), algebra.device)
        return (
            op,
            {
                "route": "public_composition" if case.family == "geometry" else "public_direct",
                "policy_selected_route": None,
                "phase_note": "No separate root request; public construction inclusive",
            },
            stages,
        )
    request, stages["request_ms"] = timed(lambda: request_for(algebra, case), algebra.device)
    router, limits = algebra._planner.router, algebra.resource_limits
    default, stages["default_planning_ms"] = timed(
        lambda: router.select(request, DEFAULT_PLANNING_POLICY, limits), algebra.device
    )
    if route == "default":
        selection = default
        stages["planning_ms"] = stages.pop("default_planning_ms")
    else:
        # Restrict only the root provider set; children keep the normal router/policy.
        providers = tuple(p for p in router.providers if p.identity == (case.family, route))
        if not providers:
            raise Unsupported(f"unknown_route: {case.family}/{route}")
        selection, stages["planning_ms"] = timed(
            lambda: type(router)(providers).select(request, ForceRoot(route), limits), algebra.device
        )
        assert selection.route == route
    kernel, stages["preparation_ms"] = timed(selection.build, algebra.device)
    ins, out = contracts(algebra, case, storage)
    method = (
        "forward_pairwise_compact"
        if case.pairwise
        else ("forward_compact" if case.family in {"product", "unary"} else "forward")
    )
    op, stages["wrapper_ms"] = timed(lambda: PlannedOperation(kernel, ins, out, method=method), algebra.device)
    metadata = selection_metadata(selection)
    metadata["policy_selected_route"] = default.route
    metadata["buffer_devices"] = sorted({str(b.device) for b in op.buffers()})
    execution_devices = {str(algebra.device), *metadata["buffer_devices"]}

    def has_cpu_bridge(node):
        return (node["family"], node["route"]) in {("action", "vector_matrix"), ("bivector_exp", "taylor")} or any(
            has_cpu_bridge(child) for child in node["children"]
        )

    if algebra.device.type == "mps" and has_cpu_bridge(metadata):
        execution_devices.add("cpu")
    metadata["execution_devices"] = sorted(execution_devices)
    return op, metadata, stages
