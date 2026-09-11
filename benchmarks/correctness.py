"""Independent CPU float64 semantics and full-output/input-VJP gates."""

import torch

from tests.helpers.small_oracle import SmallCliffordOracle

from .adapter import Unsupported, contracts, public_operation

# Fixed before measurements; never adjusted after a failed comparison.
TOLERANCES = {
    torch.float64: (1e-10, 1e-12),
    torch.float32: (1e-4, 1e-5),
    torch.float16: (5e-2, 5e-3),
    torch.bfloat16: (1e-1, 2e-2),
}


def reference(algebra, case, storage, max_terms=200_000, max_order=128):
    """Build oracle coefficients independently; avoid a Python autograd node per term."""
    oracle = SmallCliffordOracle(*case.signature)
    ins, out = contracts(algebra, case, storage)
    if max(c.layout.dim for c in (*ins, out)) > algebra.resource_limits.max_lanes:
        raise Unsupported("declared_layout_exceeds_resource_max_lanes")
    bases = tuple(c.layout.basis_indices for c in ins)
    output = out.layout.basis_indices

    def product(left_basis, right_basis, output_basis, op="geometric_product"):
        if len(left_basis) * len(right_basis) > max_terms:
            raise Unsupported("reference_budget_exceeded: independent product terms")
        positions = {b: i for i, b in enumerate(output_basis)}
        terms = [
            (i, j, positions[a ^ b], oracle.operation_coefficient(a, b, op))
            for i, a in enumerate(left_basis)
            for j, b in enumerate(right_basis)
            if a ^ b in positions and oracle.operation_coefficient(a, b, op)
        ]
        indices = torch.tensor([t[:3] for t in terms], dtype=torch.long).reshape(-1, 3)
        coefficients = torch.tensor([t[3] for t in terms], dtype=torch.float64)

        def execute(a, b):
            values = a[..., indices[:, 0]] * b[..., indices[:, 1]] * coefficients
            result = values.new_zeros(*values.shape[:-1], len(output_basis))
            return result.index_add(-1, indices[:, 2], values)

        return execute

    if case.family == "product":
        multiply = product(*bases, output, case.operation)

        def fn(a, b):
            return multiply(a.unsqueeze(-2), b.unsqueeze(-3)) if case.pairwise else multiply(a, b)
    elif case.family == "action" and case.inputs[1] == (1,):
        full = oracle.full_indices
        first, second = product(bases[1], bases[0], full), product(full, bases[1], output)

        def fn(x, normal):
            return second(first(normal, oracle.grade_involution(x, bases[0])), oracle.blade_inverse(normal, bases[1]))
    elif case.family in {"bivector_exp", "action"}:
        if (1 << (algebra.n - 1)) > max_order:
            raise Unsupported(f"reference_budget_exceeded: exponential matrix order > {max_order}")
        even = oracle.indices_for_grades(range(0, algebra.n + 1, 2))
        b_basis = bases[0] if case.family == "bivector_exp" else bases[1]
        multiply = product(b_basis, even, even)
        eye = torch.eye(len(even), dtype=torch.float64)

        def exp(b):
            matrix = multiply(b.unsqueeze(-2), eye).transpose(-1, -2)
            return torch.matrix_exp(matrix)[..., :, 0]

        if case.family == "action":
            full = oracle.full_indices
            first, second = product(even, bases[0], full), product(full, even, output)

            def fn(x, b):
                rotor = exp(-0.5 * b)
                return second(first(rotor, x), oracle.reverse(rotor, even))
        else:
            positions = {b: i for i, b in enumerate(even)}
            indices = torch.tensor([positions.get(b, 0) for b in output])
            mask = torch.tensor([float(b in positions) for b in output])

            def fn(b):
                return exp(b)[..., indices] * mask
    elif case.family == "unary":
        positions = {b: i for i, b in enumerate(bases[0])}
        indices = torch.tensor([positions.get(b, 0) for b in output])
        signs = []
        for b in output:
            g = b.bit_count()
            sign = oracle.reverse_sign(b) if case.operation in {"reverse", "clifford_conjugation"} else 1.0
            if case.operation in {"grade_involution", "clifford_conjugation"}:
                sign *= (-1.0) ** g
            signs.append(sign if b in positions else 0.0)
        coefficients = torch.tensor(signs)

        def fn(x):
            return x[..., indices] * coefficients
    elif case.family == "metric":

        def fn(x):
            return oracle.signature_norm_squared(x, bases[0])
    elif case.family == "permutation":

        def fn(x):
            return oracle.pseudoscalar_product(x, input_indices=bases[0], output_indices=output)
    elif case.family == "geometry":
        full = oracle.full_indices
        first, second = product(bases[1], bases[0], full), product(full, bases[1], output)

        def fn(x, normal):
            return second(first(normal, oracle.grade_involution(x, bases[0])), oracle.blade_inverse(normal, bases[1]))
    elif case.operation == "lane_grade_energy":

        def fn(x):
            return torch.stack(
                [
                    x[..., [i for i, b in enumerate(bases[0]) if b.bit_count() == g]].square().sum(-1)
                    for g in range(algebra.n + 1)
                ],
                -1,
            )
    else:
        multiply = product(*bases, (0,))

        def fn(a, b):
            return multiply(oracle.clifford_conjugation(a, bases[0]), b)

    def wrapped(*values):
        compact = tuple(x if storage == "compact" else x[..., list(b)] for x, b in zip(values, bases))
        result = fn(*compact)
        if storage == "canonical" and case.family != "forms":
            expanded = result.new_zeros(*result.shape[:-1], algebra.dim)
            return expanded.index_copy(-1, torch.tensor(output), result)
        return result

    return wrapped


def check(fn, expected_fn, args, backward, dtype):
    atol, rtol = TOLERANCES[dtype][1], TOLERANCES[dtype][0]
    actual_inputs = tuple(x.detach().requires_grad_(backward) for x in args)
    reference_inputs = tuple(x.detach().cpu().double().requires_grad_(backward) for x in args)
    with torch.set_grad_enabled(backward):
        expected = expected_fn(*reference_inputs)
        actual = fn(*actual_inputs)
        if actual.device != args[0].device or actual.dtype != dtype:
            raise AssertionError("output placement/dtype differs from declared workload")
        torch.testing.assert_close(actual.cpu().double(), expected, atol=atol, rtol=rtol)
        errors = {
            "output_max_abs_error": float((actual.cpu().double() - expected).detach().abs().max())
            if actual.numel()
            else 0.0
        }
        if backward:
            seed = torch.linspace(0.5, 1.5, expected.numel(), dtype=torch.float64).reshape(expected.shape)
            got = torch.autograd.grad(actual, actual_inputs, seed.to(actual), allow_unused=False)
            want = torch.autograd.grad(expected, reference_inputs, seed, allow_unused=False)
            errors["gradient_max_abs_error"] = 0.0
            for a, b in zip(got, want):
                torch.testing.assert_close(a.cpu().double(), b, atol=atol, rtol=rtol)
                if a.numel():
                    errors["gradient_max_abs_error"] = max(
                        errors["gradient_max_abs_error"], float((a.cpu().double() - b).abs().max())
                    )
    return {"status": "passed", "reference": "independent-oracle-cpu-f64", "rtol": rtol, "atol": atol, **errors}


def gate(algebra, case, storage, fn, args, backward, *, max_terms=200_000, max_order=128):
    oracle = reference(algebra, case, storage, max_terms, max_order)
    # Public boundary parity is independently gated against the same oracle.
    try:
        public = check(public_operation(algebra, case, storage), oracle, args, backward, algebra.dtype)
    except AssertionError as exc:
        public = {"status": "failed", "reason": str(exc)}
    result = check(fn, oracle, args, backward, algebra.dtype)
    result["public_default_check"] = public
    return result, oracle
