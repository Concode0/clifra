# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0
"""Concrete mathematical API over private planning and execution."""

from __future__ import annotations

import torch

from ._kernel.basis import expand_output_grades, normalize_grade_product_op
from ._kernel.device import resolve_device, resolve_dtype
from ._kernel.planning.layouts import ProductRequest
from ._kernel.planning.planner import GradePlanner
from ._kernel.planning.policy import DEFAULT_PLANNING_POLICY
from ._kernel.planning.resources import DEFAULT_RESOURCE_LIMITS
from ._kernel.planning.unary import UnaryRequest, normalize_unary_op
from .layout import AlgebraSpec, GradeLayout, Layout
from .operation import PlannedOperation
from .tensors import TensorContract

Declaration = Layout | TensorContract | None


class AlgebraContext:
    """Clifford signature and operations with explicit coefficient contracts.

    Omitted input declarations mean full-basis tensors. Layouts declare compact
    storage; TensorContract explicitly selects compact or canonical storage.
    Semantic layouts are never inferred from tensor widths.
    """

    def __init__(self, p: int, q: int = 0, r: int = 0, *, device="cpu", dtype=torch.float32, registry=None):
        self.spec = AlgebraSpec(p, q, r)
        self.p, self.q, self.r = self.spec.p, self.spec.q, self.spec.r
        self.n, self.dim = self.spec.n, self.spec.dim
        self.num_grades = self.n + 1
        self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        self._dtype = resolve_dtype(dtype)
        self._planning_policy = DEFAULT_PLANNING_POLICY
        self._resource_limits = DEFAULT_RESOURCE_LIMITS
        from .executors import ExecutorRegistry

        if registry is not None and not isinstance(registry, ExecutorRegistry):
            raise TypeError("registry must be an ExecutorRegistry")
        self._registry = ExecutorRegistry.default() if registry is None else registry
        self._planner = GradePlanner(self)
        self._sync_eps()

    @property
    def registry(self):
        """Immutable provider collection used for planning this algebra's operations."""
        return self._registry

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def layout(self, grades=None) -> GradeLayout:
        """Declare whole grades, or the full basis when omitted."""
        grades = tuple(range(self.n + 1) if grades is None else grades)
        return self._planner.layout(grades) if grades else self.spec.layout(())

    def to(self, device=None, dtype=None):
        """Set defaults for future plans; existing plans move independently."""
        if device is not None:
            self._device = torch.device(resolve_device(device) if str(device) == "auto" else device)
        if dtype is not None:
            self._dtype = resolve_dtype(dtype)
        self._sync_eps()
        self._planner.clear_cache()
        return self

    def _apply(self, fn):
        probe = fn(torch.empty((), device=self.device, dtype=self.dtype))
        return self.to(probe.device, probe.dtype)

    def _sync_eps(self):
        self.eps = float(torch.finfo(self.dtype).eps)
        self.eps_sq = self.eps**2

    def _contract(self, declaration: Declaration, *, default=None) -> TensorContract:
        if declaration is None:
            declaration = self.layout() if default is None else default
        result = declaration if isinstance(declaration, TensorContract) else TensorContract(declaration)
        if result.spec != self.spec:
            raise ValueError("tensor contract signature does not match algebra signature")
        if not isinstance(result.layout, GradeLayout):
            raise NotImplementedError("the current kernel supports GradeLayout declarations")
        return result

    @staticmethod
    def _placement(*values):
        device, dtype = values[0].device, values[0].dtype
        for value in values[1:]:
            if value.device != device:
                raise ValueError("operands must be on the same device")
            dtype = torch.promote_types(dtype, value.dtype)
        return device, dtype

    def _on(self, operation, device, dtype):
        if device == self.device and dtype == self.dtype:
            return operation
        return operation.to(device=device, dtype=dtype)

    def _build_product(
        self, *, op="geometric_product", left=None, right=None, output=None, pairwise=False, device=None, dtype=None
    ):
        left, right = self._contract(left), self._contract(right)
        op = normalize_grade_product_op(op)
        if output is None:
            full = tuple(range(self.n + 1))
            output = self.layout(
                full
                if left.grades == right.grades == full
                else expand_output_grades(left.grades, right.grades, self.n, op=op)
            )
        output = self._contract(output)
        request = ProductRequest(
            spec=self.spec,
            op=op,
            left=TensorContract(left.layout),
            right=TensorContract(right.layout),
            output=TensorContract(output.layout),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )
        kernel = self._planner.product_executor(request)
        return PlannedOperation(
            kernel, (left, right), output, method="forward_pairwise_compact" if pairwise else "forward_compact"
        )

    def plan_product(
        self,
        *,
        op="geometric_product",
        left: Declaration = None,
        right: Declaration = None,
        output: Declaration = None,
        pairwise=False,
    ) -> PlannedOperation:
        """Plan a binary product for fixed contracts and item-axis behavior."""
        return self._build_product(op=op, left=left, right=right, output=output, pairwise=pairwise)

    def product(
        self,
        a,
        b,
        *,
        op="geometric_product",
        left: Declaration = None,
        right: Declaration = None,
        output: Declaration = None,
        pairwise=False,
    ):
        """Execute a declared binary Clifford product."""
        device, dtype = self._placement(a, b)
        return self._build_product(
            op=op, left=left, right=right, output=output, pairwise=pairwise, device=device, dtype=dtype
        )(a, b)

    def geometric_product(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        """Geometric product AB."""
        return self.product(a, b, left=left, right=right, output=output, pairwise=pairwise)

    def wedge(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        return self.product(a, b, op="wedge", left=left, right=right, output=output, pairwise=pairwise)

    def left_contraction(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        return self.product(a, b, op="left_contraction", left=left, right=right, output=output, pairwise=pairwise)

    def right_contraction(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        return self.product(a, b, op="right_contraction", left=left, right=right, output=output, pairwise=pairwise)

    def symmetric_product(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        """Normalized anticommutator (AB + BA) / 2."""
        return self.product(a, b, op="symmetric_product", left=left, right=right, output=output, pairwise=pairwise)

    def commutator_product(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        """Unnormalized commutator AB - BA."""
        return self.product(a, b, op="commutator_product", left=left, right=right, output=output, pairwise=pairwise)

    def anti_commutator_product(self, a, b, *, left=None, right=None, output=None, pairwise=False):
        """Unnormalized anticommutator AB + BA."""
        return self.product(
            a, b, op="anti_commutator_product", left=left, right=right, output=output, pairwise=pairwise
        )

    def _build_unary(self, *, op, input=None, output=None, device=None, dtype=None):
        input = self._contract(input)
        if op == "grade_projection" and output is None:
            raise ValueError("grade projection requires an output declaration")
        output = self._contract(output, default=input.layout)
        request = UnaryRequest(
            spec=self.spec,
            op=normalize_unary_op(op),
            input=TensorContract(input.layout),
            output=TensorContract(output.layout),
            dtype=self.dtype if dtype is None else dtype,
            device=self.device if device is None else device,
        )
        return PlannedOperation(self._planner.unary_executor(request), (input,), output, method="forward_compact")

    def plan_unary(self, *, op, input: Declaration = None, output: Declaration = None) -> PlannedOperation:
        return self._build_unary(op=op, input=input, output=output)

    def unary(self, values, *, op, input: Declaration = None, output: Declaration = None):
        return self._build_unary(op=op, input=input, output=output, device=values.device, dtype=values.dtype)(values)

    def reverse(self, values, *, input=None, output=None):
        return self.unary(values, op="reverse", input=input, output=output)

    def grade_involution(self, values, *, input=None, output=None):
        return self.unary(values, op="grade_involution", input=input, output=output)

    def clifford_conjugation(self, values, *, input=None, output=None):
        return self.unary(values, op="clifford_conjugation", input=input, output=output)

    def grade_projection(self, values, *, output, input=None):
        return self.unary(values, op="grade_projection", input=input, output=output)

    def _build_special(self, kind, *, input=None, output=None, device=None, dtype=None):
        input = self._contract(input)
        device, dtype = (self.device if device is None else device), (self.dtype if dtype is None else dtype)
        if kind == "signature_norm_squared":
            output = self._contract(output, default=self.layout((0,)))
            if output.layout.grades != (0,):
                raise ValueError("signature norm output must be scalar")
            kernel = self._planner.signature_norm_squared_executor(
                input_layout=input.layout, device=device, dtype=dtype
            )
        elif kind == "pseudoscalar_product":
            output = self._contract(output, default=self.layout(self.n - grade for grade in input.grades))
            kernel = self._planner.pseudoscalar_product_executor(
                input_layout=input.layout, output_layout=output.layout, device=device, dtype=dtype
            )
        else:
            if self.n < 2:
                raise ValueError("bivector exp requires at least two basis vectors")
            full_grades = tuple(range(self.n + 1))
            if input.grades != full_grades and input.grades != (2,):
                raise ValueError("bivector exp input must declare grade 2 or the full basis")
            if output is None:
                output = input.layout if input.grades == full_grades else self.layout(range(0, self.n + 1, 2))
            output = self._contract(output)
            kernel = self._planner.bivector_exp_executor(
                input_layout=self.layout((2,)), output_layout=output.layout, device=device, dtype=dtype
            )
            if input.grades != (2,):
                projection = self._build_unary(
                    op="grade_projection", input=input.layout, output=self.layout((2,)), device=device, dtype=dtype
                )
                kernel = torch.nn.Sequential(projection, kernel)
        return PlannedOperation(kernel, (input,), output)

    def plan_signature_norm_squared(self, *, input=None, output=None):
        return self._build_special("signature_norm_squared", input=input, output=output)

    def signature_norm_squared(self, values, *, input=None, output=None):
        """Signed scalar form <A reverse(A)>_0."""
        return self._build_special(
            "signature_norm_squared", input=input, output=output, device=values.device, dtype=values.dtype
        )(values)

    def plan_pseudoscalar_product(self, *, input=None, output=None):
        return self._build_special("pseudoscalar_product", input=input, output=output)

    def pseudoscalar_product(self, values, *, input=None, output=None):
        return self._build_special(
            "pseudoscalar_product", input=input, output=output, device=values.device, dtype=values.dtype
        )(values)

    def plan_bivector_exp(self, *, input=None, output=None):
        return self._build_special("bivector_exp", input=input, output=output)

    def bivector_exp(self, values, *, input=None, output=None):
        """Exponentiate grade 2; full-basis input is projected to grade 2."""
        return self._build_special(
            "bivector_exp", input=input, output=output, device=values.device, dtype=values.dtype
        )(values)

    def plan_versor_action(self, *, grade, input=None, parameter=None, output=None):
        """Plan reflection (grade 1) or exp(-B/2) rotor action (grade 2)."""

        input = self._contract(input)
        parameter = self._contract(parameter)
        if parameter.grades != (grade,):
            raise ValueError("versor parameter must explicitly declare its parameter grade")
        output = self._contract(output, default=input.layout)
        kernel = self._planner.action_executor(
            "versor",
            grade=grade,
            input_layout=input.layout,
            parameter_layout=parameter.layout,
            output_layout=output.layout,
        )
        return PlannedOperation(kernel, (input, parameter), output)

    def versor_action(self, values, weights, *, grade, input=None, parameter=None, output=None):
        device, dtype = self._placement(values, weights)
        return self._on(
            self.plan_versor_action(grade=grade, input=input, parameter=parameter, output=output), device, dtype
        )(values, weights)

    def _build_geometry(self, kind, *, input=None, left=None, right=None, output=None):
        from ._kernel.composition import Geometry

        declarations = (
            (input,) if kind == "blade_inverse" else (left, input, right) if kind == "sandwich" else (input, right)
        )
        inputs = tuple(self._contract(item) for item in declarations)
        default = inputs[1].layout if kind == "sandwich" else inputs[0].layout
        output = self._contract(output, default=default)
        kernel = Geometry(self, kind, tuple(item.layout for item in inputs), output.layout)
        return PlannedOperation(kernel, inputs, output)

    def plan_blade_inverse(self, *, input=None, output=None):
        return self._build_geometry("blade_inverse", input=input, output=output)

    def blade_inverse(self, a, *, input=None, output=None):
        return self._on(self.plan_blade_inverse(input=input, output=output), a.device, a.dtype)(a)

    def plan_blade_project(self, *, input=None, blade=None, output=None):
        return self._build_geometry("blade_project", input=input, right=blade, output=output)

    def blade_project(self, values, a, *, input=None, blade=None, output=None):
        device, dtype = self._placement(values, a)
        return self._on(self.plan_blade_project(input=input, blade=blade, output=output), device, dtype)(values, a)

    def plan_blade_reject(self, *, input=None, blade=None, output=None):
        return self._build_geometry("blade_reject", input=input, right=blade, output=output)

    def blade_reject(self, values, a, *, input=None, blade=None, output=None):
        device, dtype = self._placement(values, a)
        return self._on(self.plan_blade_reject(input=input, blade=blade, output=output), device, dtype)(values, a)

    def plan_reflect(self, *, input=None, normal=None, output=None):
        return self._build_geometry("versor", input=input, right=normal, output=output)

    def reflect(self, values, a, *, input=None, normal=None, output=None):
        device, dtype = self._placement(values, a)
        return self._on(self.plan_reflect(input=input, normal=normal, output=output), device, dtype)(values, a)

    def plan_versor_product(self, *, input=None, versor=None, output=None):
        return self._build_geometry("versor", input=input, right=versor, output=output)

    def versor_product(self, a, values, *, input=None, versor=None, output=None):
        device, dtype = self._placement(values, a)
        return self._on(self.plan_versor_product(input=input, versor=versor, output=output), device, dtype)(values, a)

    def plan_sandwich_action(self, *, left=None, input=None, right=None, output=None):
        """Plan L X R with ordinary tensor broadcasting."""
        return self._build_geometry("sandwich", left=left, input=input, right=right, output=output)

    def sandwich_product(self, a, values, b, *, left=None, input=None, right=None, output=None):
        device, dtype = self._placement(a, values, b)
        return self._on(self.plan_sandwich_action(left=left, input=input, right=right, output=output), device, dtype)(
            a, values, b
        )

    def plan_linear_action(self, *, input=None, output=None):
        """Plan induced action of [channels, n, n] matrices on [..., channels, lanes]."""

        input = self._contract(input)
        output = self._contract(output, default=input.layout)
        self._planner.linear_action_plan(input_layout=input.layout, output_layout=output.layout)
        kernel = self._planner.action_executor(
            "linear",
            input_layout=input.layout,
            output_layout=output.layout,
        )
        return PlannedOperation(kernel, (input, None), output)

    def linear_action(self, values, matrix, *, input=None, output=None):
        return self.plan_linear_action(input=input, output=output).to(device=values.device)(values, matrix)

    def scalar_product(self, a, b, *, left=None, right=None):
        return self.geometric_product(a, b, left=left, right=right, output=self.layout((0,)))

    def _aligned(self, a, b, left, right):
        self._placement(a, b)
        left, right = self._contract(left), self._contract(right)
        layout = left.layout.union(right.layout)
        return (
            layout.convert(left.to_compact(a), left.layout),
            layout.convert(right.to_compact(b), right.layout),
            layout,
        )

    def lane_dot_product(self, a, b, *, left=None, right=None):
        a, b, _ = self._aligned(a, b, left, right)
        return (a * b).sum(dim=-1, keepdim=True)

    def lane_energy(self, values, *, input=None, grades=None):
        """Coefficient energy, optionally selected by grade, without batch reduction."""
        contract = self._contract(input)
        values = contract.to_compact(values)
        if grades is not None:
            values = values * contract.layout.grade_mask(grades, device=values.device)
        return values.square().sum(dim=-1, keepdim=True)

    def lane_norm(self, values, *, input=None):
        return self.lane_energy(values, input=input).sqrt()

    def lane_distance(self, a, b, *, left=None, right=None):
        a, b, _ = self._aligned(a, b, left, right)
        return (a - b).square().sum(dim=-1, keepdim=True).sqrt()

    def lane_grade_energy(self, values, *, input=None):
        contract = self._contract(input)
        values = contract.to_compact(values)
        result = values.new_zeros(*values.shape[:-1], self.n + 1)
        indices = contract.layout.grade_indices_tensor(device=values.device).expand_as(values)
        return result.scatter_add(-1, indices, values.square())

    def lane_grade_norms(self, values, *, input=None):
        return self.lane_grade_energy(values, input=input).sqrt()

    def lane_grade_distribution(self, values, *, input=None, eps=1.0e-8):
        energy = self.lane_grade_energy(values, input=input)
        return energy / (energy.sum(dim=-1, keepdim=True) + eps)

    def conjugate_scalar_form(self, a, b, *, left=None, right=None):
        from ._kernel.forms import conjugate_scalar_form_signs

        a, b, layout = self._aligned(a, b, left, right)
        signs = conjugate_scalar_form_signs(
            self, layout=layout, device=a.device, dtype=torch.promote_types(a.dtype, b.dtype)
        )
        return (signs * a * b).sum(dim=-1, keepdim=True)

    def format_multivector(self, values, **kwargs):
        from .formatting import format_multivector

        return format_multivector(self, values, **kwargs)
