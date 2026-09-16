"""Test-only independent references for routed benchmark workloads."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator

import torch

from benchmarks.cases import BenchmarkCase, ExecutionPlacement, LayoutCase

# Keep temporary reference terms bounded independently of caller batch size.
_MAX_INTERACTION_ROWS = 131_072
_MAX_TERM_ELEMENTS = 4_194_304


def _basis_indices(n: int, grades: tuple[int, ...]) -> tuple[int, ...]:
    """Return only declared basis masks, without scanning the full 2**n space."""

    masks = []
    for grade in grades:
        masks.extend(sum(1 << bit for bit in combination) for combination in itertools.combinations(range(n), grade))
    return tuple(sorted(masks))


def _geometric_coefficient(left: int, right: int, signature: tuple[int, int, int]) -> int:
    """Return one independently derived basis-blade product coefficient."""

    p, q, _ = signature
    swaps = sum((right & ((1 << bit) - 1)).bit_count() for bit in range(sum(signature)) if left & (1 << bit))
    coefficient = -1 if swaps % 2 else 1
    overlap = left & right
    negative_mask = sum(1 << bit for bit in range(p, p + q))
    null_mask = sum(1 << bit for bit in range(p + q, sum(signature)))
    if overlap & null_mask:
        return 0
    if (overlap & negative_mask).bit_count() % 2:
        coefficient = -coefficient
    return coefficient


def _operation_coefficient(left: int, right: int, case: BenchmarkCase) -> float:
    operation = case.operation
    geometric = _geometric_coefficient(left, right, case.signature)
    output_grade = (left ^ right).bit_count()
    left_grade, right_grade = left.bit_count(), right.bit_count()
    if operation == "geometric_product":
        return geometric
    if operation == "wedge":
        return geometric if not (left & right) else 0
    if operation == "left_contraction":
        return geometric if left_grade <= right_grade and output_grade == right_grade - left_grade else 0
    if operation == "right_contraction":
        return geometric if right_grade <= left_grade and output_grade == left_grade - right_grade else 0
    reverse = _geometric_coefficient(right, left, case.signature)
    if operation == "symmetric_product":
        return 0.5 * (geometric + reverse)
    if operation == "commutator_product":
        return geometric - reverse
    if operation == "anti_commutator_product":
        return geometric + reverse
    raise ValueError(f"unsupported product operation {operation!r}")


def _bit_count(values: torch.Tensor, n: int) -> torch.Tensor:
    counts = torch.zeros_like(values)
    for bit in range(n):
        counts += torch.bitwise_and(values, 1 << bit).ne(0)
    return counts


def _geometric_coefficients(
    left: torch.Tensor,
    right: torch.Tensor,
    signature: tuple[int, int, int],
) -> torch.Tensor:
    """Vectorized independent Clifford coefficient calculation on CPU masks."""

    p, q, r = signature
    n = p + q + r
    parity = torch.zeros_like(left, dtype=torch.bool)
    for bit in range(n):
        left_has_bit = torch.bitwise_and(left, 1 << bit).ne(0)
        lower_right = torch.bitwise_and(right, (1 << bit) - 1)
        parity ^= left_has_bit & _bit_count(lower_right, bit).bitwise_and(1).bool()
    overlap = torch.bitwise_and(left, right)
    if q:
        negative_mask = sum(1 << bit for bit in range(p, p + q))
        parity ^= _bit_count(torch.bitwise_and(overlap, negative_mask), n).bitwise_and(1).bool()
    coefficients = torch.where(parity, -torch.ones_like(left), torch.ones_like(left))
    if r:
        null_mask = sum(1 << bit for bit in range(p + q, n))
        coefficients = torch.where(torch.bitwise_and(overlap, null_mask).ne(0), 0, coefficients)
    return coefficients


def _operation_coefficients(
    left: torch.Tensor,
    right: torch.Tensor,
    case: BenchmarkCase,
) -> torch.Tensor:
    geometric = _geometric_coefficients(left, right, case.signature)
    output = torch.bitwise_xor(left, right)
    overlap = torch.bitwise_and(left, right)
    n = sum(case.signature)
    left_grade = _bit_count(left, n)
    right_grade = _bit_count(right, n)
    output_grade = _bit_count(output, n)
    operation = case.operation
    if operation == "geometric_product":
        return geometric
    if operation == "wedge":
        return torch.where(overlap.eq(0), geometric, 0)
    if operation == "left_contraction":
        valid = (left_grade <= right_grade) & (output_grade == right_grade - left_grade)
        return torch.where(valid, geometric, 0)
    if operation == "right_contraction":
        valid = (right_grade <= left_grade) & (output_grade == left_grade - right_grade)
        return torch.where(valid, geometric, 0)
    reverse = _geometric_coefficients(right, left, case.signature)
    if operation == "symmetric_product":
        return (geometric + reverse).to(torch.float64) * 0.5
    if operation == "commutator_product":
        return geometric - reverse
    if operation == "anti_commutator_product":
        return geometric + reverse
    raise ValueError(f"unsupported product operation {operation!r}")


def _physical_positions(basis: torch.Tensor, storage: str) -> torch.Tensor:
    return basis if storage == "canonical" else torch.arange(basis.numel(), dtype=torch.long)


def _interaction_chunks(case: BenchmarkCase, *, max_rows: int) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield independently derived sparse interaction tuples in bounded chunks."""

    n = sum(case.signature)
    left_basis = torch.tensor(_basis_indices(n, case.inputs[0].grades), dtype=torch.long)
    right_basis = torch.tensor(_basis_indices(n, case.inputs[1].grades), dtype=torch.long)
    output_basis = torch.tensor(_basis_indices(n, case.output.grades), dtype=torch.long)
    left_positions = _physical_positions(left_basis, case.inputs[0].storage)
    right_positions = _physical_positions(right_basis, case.inputs[1].storage)
    output_positions = _physical_positions(output_basis, case.output.storage)
    total = left_basis.numel() * right_basis.numel()
    if total == 0 or output_basis.numel() == 0:
        return
    for start in range(0, total, max_rows):
        flat = torch.arange(start, min(start + max_rows, total), dtype=torch.long)
        left_ordinal = torch.div(flat, right_basis.numel(), rounding_mode="floor")
        right_ordinal = flat.remainder(right_basis.numel())
        left_masks = left_basis[left_ordinal]
        right_masks = right_basis[right_ordinal]
        outputs = torch.bitwise_xor(left_masks, right_masks)
        positions = torch.searchsorted(output_basis, outputs)
        clamped = positions.clamp_max(output_basis.numel() - 1)
        valid = (positions < output_basis.numel()) & output_basis[clamped].eq(outputs)
        coefficients = _operation_coefficients(left_masks, right_masks, case)
        valid &= coefficients.ne(0)
        yield (
            left_positions[left_ordinal][valid],
            right_positions[right_ordinal][valid],
            output_positions[positions[valid]],
            coefficients[valid].to(torch.float64),
        )


def _output_lane_dim(case: BenchmarkCase) -> int:
    if case.output.storage == "canonical":
        return 1 << sum(case.signature)
    return len(_basis_indices(sum(case.signature), case.output.grades))


def _chunk_rows(values: tuple[torch.Tensor, torch.Tensor]) -> int:
    leading = torch.broadcast_shapes(values[0].shape[:-1], values[1].shape[:-1])
    batch_elements = math.prod(leading) if leading else 1
    return max(1, min(_MAX_INTERACTION_ROWS, _MAX_TERM_ELEMENTS // batch_elements))


def reference_product(values: tuple[torch.Tensor, torch.Tensor], case: BenchmarkCase) -> torch.Tensor:
    """Evaluate a product from independent sparse interactions on CPU float64."""

    left, right = (value.to(device="cpu").to(dtype=torch.float64) for value in values)
    leading = torch.broadcast_shapes(left.shape[:-1], right.shape[:-1])
    result = left.new_zeros(*leading, _output_lane_dim(case))
    for left_positions, right_positions, output_positions, coefficients in _interaction_chunks(
        case, max_rows=_chunk_rows((left, right))
    ):
        terms = left[..., left_positions] * right[..., right_positions] * coefficients
        result = result.index_add(-1, output_positions, terms)
    return result


def reference_product_with_gradients(
    values: tuple[torch.Tensor, torch.Tensor], case: BenchmarkCase
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Return output and analytical squared-output-loss input gradients."""

    left, right = (value.detach().to(device="cpu").to(dtype=torch.float64) for value in values)
    output = reference_product((left, right), case)
    leading = output.shape[:-1]
    left_gradient = left.new_zeros(*leading, left.shape[-1])
    right_gradient = right.new_zeros(*leading, right.shape[-1])
    for left_positions, right_positions, output_positions, coefficients in _interaction_chunks(
        case, max_rows=_chunk_rows((left, right))
    ):
        output_adjoint = 2 * output[..., output_positions] * coefficients
        left_terms = output_adjoint * right[..., right_positions]
        right_terms = output_adjoint * left[..., left_positions]
        left_gradient = left_gradient.index_add(-1, left_positions, left_terms)
        right_gradient = right_gradient.index_add(-1, right_positions, right_terms)
    return output, (left_gradient.sum_to_size(left.shape), right_gradient.sum_to_size(right.shape))


def _semantic_full(value: torch.Tensor, declaration, n: int) -> torch.Tensor:
    """Legacy dense expansion retained only for small-reference cross-checks."""

    value = value.to(device="cpu").to(dtype=torch.float64)
    selected = _basis_indices(n, declaration.grades)
    zero = value.sum(dim=-1) * 0
    if declaration.storage == "canonical":
        lanes = [value[..., index] if index in selected else zero for index in range(1 << n)]
        return torch.stack(lanes, dim=-1)
    positions = {basis: position for position, basis in enumerate(selected)}
    lanes = [value[..., positions[index]] if index in positions else zero for index in range(1 << n)]
    return torch.stack(lanes, dim=-1)


def dense_reference_product(values: tuple[torch.Tensor, torch.Tensor], case: BenchmarkCase) -> torch.Tensor:
    """Original simple formulation, reserved for tests on small algebras."""

    n = sum(case.signature)
    left = _semantic_full(values[0], case.inputs[0], n)
    right = _semantic_full(values[1], case.inputs[1], n)
    zero = left[..., 0] * 0 + right[..., 0] * 0
    lanes = [zero for _ in range(1 << n)]
    for left_index in _basis_indices(n, case.inputs[0].grades):
        for right_index in _basis_indices(n, case.inputs[1].grades):
            coefficient = _operation_coefficient(left_index, right_index, case)
            if coefficient:
                output_index = left_index ^ right_index
                lanes[output_index] = (
                    lanes[output_index] + coefficient * left[..., left_index] * right[..., right_index]
                )
    full = torch.stack(lanes, dim=-1)
    if case.output.storage == "canonical":
        allowed = set(_basis_indices(n, case.output.grades))
        zero = full.sum(dim=-1) * 0
        return torch.stack([full[..., index] if index in allowed else zero for index in range(1 << n)], -1)
    return full[..., list(_basis_indices(n, case.output.grades))]


def _tolerances(family: str, dtype: str) -> tuple[float, float]:
    # Qualification compares float32 device reductions with a float64 oracle.
    # The tolerance admits ordinary reduction-order error at hundreds of terms
    # while remaining tight enough to expose sign, lane, or metric mistakes.
    if dtype == "float32":
        return (2e-3, 2e-3) if family in {"bivector_exp", "action"} else (1e-3, 1e-3)
    return (2e-9, 2e-9) if family in {"bivector_exp", "action"} else (1e-10, 1e-10)


def qualify_product(
    operation,
    values,
    case: BenchmarkCase,
    placement: ExecutionPlacement = ExecutionPlacement(),
    *,
    check_gradients: bool,
) -> dict[str, object]:
    """Gate timing on forward output and, when requested, both input gradients."""

    actual_inputs = tuple(value.detach().clone().requires_grad_(check_gradients) for value in values)
    atol, rtol = _tolerances(case.family, placement.dtype)
    try:
        actual = operation(*actual_inputs)
        if check_gradients:
            expected, expected_gradients = reference_product_with_gradients(values, case)
        else:
            expected = reference_product(values, case)
            expected_gradients = ()
        torch.testing.assert_close(actual.detach().cpu().double(), expected, atol=atol, rtol=rtol)
        max_abs_error = float((actual.detach().cpu().double() - expected).abs().max().item())
        if check_gradients:
            actual_gradients = torch.autograd.grad(actual.square().sum(), actual_inputs)
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(
                    actual_gradient.detach().cpu().double(), expected_gradient, atol=atol, rtol=rtol
                )
        return {
            "status": "passed",
            "reference": "tests/benchmarks/correctness_reference.py:reference_product",
            "claim": "independent sparse Clifford basis interactions; forward and squared-loss input gradients",
            "gradient_checked": check_gradients,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs_error,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "reference": "tests/benchmarks/correctness_reference.py:reference_product",
            "claim": "independent sparse Clifford basis interactions; forward and squared-loss input gradients",
            "gradient_checked": check_gradients,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _compact_semantic(value: torch.Tensor, declaration: LayoutCase, n: int) -> torch.Tensor:
    value = value.to(device="cpu", dtype=torch.float64)
    if declaration.storage == "compact":
        return value
    indices = torch.tensor(_basis_indices(n, declaration.grades), dtype=torch.long)
    return torch.index_select(value, -1, indices)


def _format_reference_output(value: torch.Tensor, declaration: LayoutCase, n: int) -> torch.Tensor:
    if declaration.storage == "compact":
        return value
    output = value.new_zeros(*value.shape[:-1], 1 << n)
    indices = torch.tensor(_basis_indices(n, declaration.grades), dtype=torch.long)
    return output.index_copy(-1, indices, value)


def _reference_structured_exp(bivector, case, even_grades):
    """Closed product of declared disjoint simple-plane exponentials."""

    n = sum(case.signature)
    positions = {blade: position for position, blade in enumerate(_basis_indices(n, (2,)))}
    if case.generator_structure == "null_plane":
        blades = ((1 << 0) | (1 << (n - 1)),)
    elif case.generator_structure == "simple_plane":
        blades = (3,)
    else:
        blades = tuple(3 << (2 * plane) for plane in range(n // 2))
    result = bivector.new_ones(*bivector.shape[:-1], 1)
    result_grades = (0,)
    for blade in blades:
        coefficient = bivector[..., positions[blade]]
        square = _geometric_coefficient(blade, blade, case.signature)
        if square < 0:
            scalar, bivector_coefficient = coefficient.cos(), coefficient.sin()
        elif square > 0:
            scalar, bivector_coefficient = coefficient.cosh(), coefficient.sinh()
        else:
            scalar, bivector_coefficient = torch.ones_like(coefficient), coefficient
        factor = torch.zeros(*coefficient.shape, len(_basis_indices(n, (0, 2))), dtype=coefficient.dtype)
        factor[..., 0] = scalar
        factor[..., 1 + positions[blade]] = bivector_coefficient
        output_grades = tuple(range(0, min(n, max(result_grades) + 2) + 1, 2))
        result = _reference_geometric(result, factor, case.signature, result_grades, (0, 2), output_grades)
        result_grades = output_grades
    if result_grades != even_grades:
        source = {blade: position for position, blade in enumerate(_basis_indices(n, result_grades))}
        zero = result.sum(-1) * 0.0
        result = torch.stack(
            [result[..., source[blade]] if blade in source else zero for blade in _basis_indices(n, even_grades)], -1
        )
    return result


def reference_bivector_exp(values: tuple[torch.Tensor, ...], case: BenchmarkCase) -> torch.Tensor:
    """Dense left-regular matrix exponential built from independent products.

    The Clifford operator construction is independent. ``torch.matrix_exp`` is
    the numerical oracle, so this does not independently validate that same
    numerical primitive inside the ``left_matrix_exp`` route.
    """

    n = sum(case.signature)
    bivector = _compact_semantic(values[0], case.inputs[0], n)
    even_grades = tuple(range(0, n + 1, 2))
    even_basis = _basis_indices(n, even_grades)
    if n > 8:
        if case.generator_structure not in {"simple_plane", "commuting_planes", "null_plane"}:
            raise ValueError("dense independent exponential reference is limited to n<=8 for random generators")
        even = _reference_structured_exp(bivector, case, even_grades)
    else:
        identity = torch.eye(len(even_basis), dtype=torch.float64)
        product_case = BenchmarkCase(
            case_id="reference.exp.left-regular",
            family="product",
            operation="geometric_product",
            signature=case.signature,
            inputs=(LayoutCase((2,)), LayoutCase(even_grades)),
            output=LayoutCase(even_grades),
        )
        columns = reference_product((bivector.unsqueeze(-2), identity), product_case)
        operator = columns.transpose(-1, -2)
        even = torch.matrix_exp(operator)[..., :, even_basis.index(0)]
    positions = {blade: position for position, blade in enumerate(even_basis)}
    zero = even.sum(-1) * 0.0
    compact = torch.stack(
        [
            even[..., positions[blade]] if blade in positions else zero
            for blade in _basis_indices(n, case.output.grades)
        ],
        dim=-1,
    )
    return _format_reference_output(compact, case.output, n)


def _lifted_action(source, matrix, input_basis, output_basis, n):
    """Vectorized independent exterior-power lift by Leibniz expansion."""

    flat = matrix.sum(dim=(-2, -1)).unsqueeze(-1) * matrix.new_zeros(len(input_basis) * len(output_basis))
    for grade in range(n + 1):
        outputs = [(position, blade) for position, blade in enumerate(output_basis) if blade.bit_count() == grade]
        inputs = [(position, blade) for position, blade in enumerate(input_basis) if blade.bit_count() == grade]
        if not outputs or not inputs:
            continue
        flat_positions = torch.tensor(
            [
                output_position * len(input_basis) + input_position
                for output_position, _ in outputs
                for input_position, _ in inputs
            ]
        )
        if grade == 0:
            coefficients = matrix.new_ones(*matrix.shape[:-2], len(flat_positions))
        else:
            row_axes = torch.tensor(
                [[bit for bit in range(n) if blade & (1 << bit)] for _, blade in outputs], dtype=torch.long
            ).repeat_interleave(len(inputs), dim=0)
            column_axes = torch.tensor(
                [[bit for bit in range(n) if blade & (1 << bit)] for _, blade in inputs], dtype=torch.long
            ).repeat((len(outputs), 1))
            terms = []
            for permutation in itertools.permutations(range(grade)):
                inversions = sum(
                    left > right for position, left in enumerate(permutation) for right in permutation[position + 1 :]
                )
                term = matrix.new_ones(*matrix.shape[:-2], len(flat_positions))
                for row_position, column_position in enumerate(permutation):
                    term = term * matrix[..., row_axes[:, row_position], column_axes[:, column_position]]
                terms.append((-1.0 if inversions % 2 else 1.0) * term)
            coefficients = torch.stack(terms, -1).sum(-1)
        flat = flat.index_copy(-1, flat_positions, coefficients)
    lifted = flat.reshape(*matrix.shape[:-2], len(output_basis), len(input_basis))
    return torch.matmul(lifted, source.unsqueeze(-1)).squeeze(-1)


def reference_linear_action(values: tuple[torch.Tensor, ...], case: BenchmarkCase) -> torch.Tensor:
    """Lift a vector map by independently expanded Leibniz determinants."""

    n = sum(case.signature)
    source = _compact_semantic(values[0], case.inputs[0], n)
    matrix = values[1].to(device="cpu", dtype=torch.float64)
    input_basis = _basis_indices(n, case.inputs[0].grades)
    output_basis = _basis_indices(n, case.output.grades)
    compact = _lifted_action(source, matrix, input_basis, output_basis, n)
    return _format_reference_output(compact, case.output, n)


def _reference_geometric(
    left,
    right,
    signature,
    left_grades,
    right_grades,
    output_grades,
):
    product_case = BenchmarkCase(
        case_id="reference.action.product",
        family="product",
        operation="geometric_product",
        signature=signature,
        inputs=(LayoutCase(left_grades), LayoutCase(right_grades)),
        output=LayoutCase(output_grades),
    )
    return reference_product((left, right), product_case)


def reference_sandwich_action(values: tuple[torch.Tensor, ...], case: BenchmarkCase) -> torch.Tensor:
    """Independent two-product sandwich for separately declared layouts."""

    n = sum(case.signature)
    left, target, right = (_compact_semantic(value, declaration, n) for value, declaration in zip(values, case.inputs))
    full_grades = tuple(range(n + 1))
    middle = _reference_geometric(
        left,
        target,
        case.signature,
        case.inputs[0].grades,
        case.inputs[1].grades,
        full_grades,
    )
    compact = _reference_geometric(
        middle,
        right,
        case.signature,
        full_grades,
        case.inputs[2].grades,
        case.output.grades,
    )
    return _format_reference_output(compact, case.output, n)


def reference_versor_action(values: tuple[torch.Tensor, ...], case: BenchmarkCase) -> torch.Tensor:
    """Independent exp/product sandwich for rotor and reflection-like actions."""

    n = sum(case.signature)
    target = _compact_semantic(values[0], case.inputs[0], n)
    parameter = _compact_semantic(values[1], case.inputs[1], n)
    full_grades = tuple(range(n + 1))
    if case.action_grade == 2 and n <= 6:
        exp_case = BenchmarkCase(
            case_id="reference.action.exp",
            family="bivector_exp",
            operation="bivector_exp",
            signature=case.signature,
            inputs=(LayoutCase((2,), leading_shape=case.inputs[1].leading_shape),),
            output=LayoutCase(tuple(range(0, n + 1, 2))),
        )
        left = reference_bivector_exp((-0.5 * parameter,), exp_case)
        rotor_grades = exp_case.output.grades
        signs = left.new_tensor(
            [(-1.0) ** (blade.bit_count() * (blade.bit_count() - 1) // 2) for blade in _basis_indices(n, rotor_grades)]
        )
        right = left * signs
        middle = _reference_geometric(left, target, case.signature, rotor_grades, case.inputs[0].grades, full_grades)
        compact = _reference_geometric(middle, right, case.signature, full_grades, rotor_grades, case.output.grades)
    elif case.action_grade == 2:
        vector_grades = (1,)
        identity = torch.eye(n, dtype=torch.float64)
        left = _reference_geometric(
            parameter.unsqueeze(-2), identity, case.signature, (2,), vector_grades, vector_grades
        )
        right = _reference_geometric(
            identity, parameter.unsqueeze(-2), case.signature, vector_grades, (2,), vector_grades
        )
        generator = (-0.5 * (left - right)).transpose(-1, -2)
        matrix = torch.matrix_exp(generator)
        # Call the independent lift core directly; the matrix is already the
        # mathematically derived vector-space action.
        input_basis = _basis_indices(n, case.inputs[0].grades)
        output_basis = _basis_indices(n, case.output.grades)
        compact = _lifted_action(target, matrix, input_basis, output_basis, n)
    else:
        metric = parameter.new_tensor(
            [1.0] * case.signature[0] + [-1.0] * case.signature[1] + [0.0] * case.signature[2]
        )
        denominator = (parameter.square() * metric).sum(-1, keepdim=True)
        eps = torch.finfo(parameter.dtype).eps ** 2
        sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
        inverse = parameter / (sign * denominator.abs().clamp_min(eps))
        involution_signs = target.new_tensor(
            [(-1.0) ** blade.bit_count() for blade in _basis_indices(n, case.inputs[0].grades)]
        )
        middle = _reference_geometric(
            parameter,
            target * involution_signs,
            case.signature,
            (1,),
            case.inputs[0].grades,
            full_grades,
        )
        compact = _reference_geometric(middle, inverse, case.signature, full_grades, (1,), case.output.grades)
    return _format_reference_output(compact, case.output, n)


def _qualification_description(case):
    if case.family == "bivector_exp":
        claim = (
            "tests/benchmarks/correctness_reference.py:reference_bivector_exp",
            "independent Clifford operator construction with torch.matrix_exp; does not independently validate "
            "the left_matrix_exp route's use of torch.matrix_exp",
        )
        if sum(case.signature) > 8:
            claim = (
                claim[0],
                "forward-only closed product of commuting simple-plane exponentials; limited to declared "
                "structured generators",
            )
        return claim
    if case.operation == "linear":
        return (
            "tests/benchmarks/correctness_reference.py:reference_linear_action",
            "independent Leibniz determinant lift; forward and squared-loss gradients",
        )
    if case.operation == "sandwich":
        return (
            "tests/benchmarks/correctness_reference.py:reference_sandwich_action",
            "independent two-product Clifford sandwich; forward and squared-loss gradients",
        )
    claim = (
        "tests/benchmarks/correctness_reference.py:reference_versor_action",
        "independent dense exp and Clifford sandwich; forward and squared-loss gradients",
    )
    if sum(case.signature) > 6 and case.action_grade == 2:
        claim = (
            claim[0],
            "independently derived vector commutator generator and Leibniz grade lift; shares torch.matrix_exp",
        )
    return claim


def qualify_case(operation, values, output_contract, case, placement, *, check_gradients):
    """Dispatch one family oracle and gate timing on contracts, output, and gradients."""

    if case.family == "product":
        return qualify_product(operation, values, case, placement, check_gradients=check_gradients)
    reference_name, claim = _qualification_description(case)
    reference = (
        reference_bivector_exp
        if case.family == "bivector_exp"
        else (
            reference_linear_action
            if case.operation == "linear"
            else reference_sandwich_action
            if case.operation == "sandwich"
            else reference_versor_action
        )
    )
    actual_inputs = tuple(value.detach().clone().requires_grad_(check_gradients) for value in values)
    reference_inputs = tuple(value.detach().cpu().double().requires_grad_(check_gradients) for value in values)
    atol, rtol = _tolerances(case.family, placement.dtype)
    try:
        actual = operation(*actual_inputs)
        output_contract.validate(actual, name="benchmark output")
        if actual.dtype != torch_dtype_for_name(placement.dtype) or actual.device.type != placement.device:
            raise AssertionError(
                f"output placement {(actual.device.type, actual.dtype)} does not match "
                f"{(placement.device, torch_dtype_for_name(placement.dtype))}"
            )
        expected = reference(reference_inputs, case)
        torch.testing.assert_close(actual.detach().cpu().double(), expected.detach(), atol=atol, rtol=rtol)
        max_abs_error = float((actual.detach().cpu().double() - expected.detach()).abs().max().item())
        if check_gradients:
            actual_gradients = torch.autograd.grad(actual.square().sum(), actual_inputs)
            expected_gradients = torch.autograd.grad(expected.square().sum(), reference_inputs)
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(
                    actual_gradient.detach().cpu().double(), expected_gradient.detach(), atol=atol * 5, rtol=rtol * 5
                )
        return {
            "status": "passed",
            "reference": reference_name,
            "claim": claim,
            "gradient_checked": check_gradients,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": max_abs_error,
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "reference": reference_name,
            "claim": claim,
            "gradient_checked": check_gradients,
            "atol": atol,
            "rtol": rtol,
            "max_abs_error": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def torch_dtype_for_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]
