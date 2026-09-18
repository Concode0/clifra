# Compose explicit rotor operations

Materialize rotors when their coefficients need to be stored, combined, or
used in further products. If $R_1$ acts first and $R_2$ second, the composite
is $R_2R_1$. Order matters: to reproduce sequential actions, multiply the
rotors directly. Bivectors in different planes generally do not commute, so
exponentiating their sum yields a different transformation.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
even = algebra.layout((0, 2))
exp = algebra.plan_bivector_exp(input=bivectors, output=even)
multiply = algebra.plan_product(left=even, right=even, output=even)
reverse = algebra.plan_unary(op="reverse", input=even)
sandwich = algebra.plan_sandwich_action(left=even, input=vectors, right=even, output=vectors)
action = algebra.plan_versor_action(grade=2, input=vectors, parameter=bivectors)
B1 = torch.tensor([0.4, 0., 0.], dtype=torch.float64)
B2 = torch.tensor([0., 0., -0.7], dtype=torch.float64)
R1, R2 = exp(-0.5 * B1), exp(-0.5 * B2)
R = multiply(R2, R1)
x = torch.tensor([1., 2., 3.], dtype=torch.float64)
composed = sandwich(R, x, reverse(R))
sequential = action(action(x, B1), B2)
torch.testing.assert_close(composed, sequential)
unit = multiply(R, reverse(R))
torch.testing.assert_close(unit, torch.tensor([1., 0., 0., 0.], dtype=x.dtype))
```

For a rotor produced by exponentiating a real bivector, reversal equals its
inverse mathematically. Reversal is not an inverse for an arbitrary even
coefficient tensor. A learned unconstrained tensor may not be a rotor, and
dividing by its coefficient norm does not enforce the rotor constraints.

`plan_sandwich_action` computes $LXR$ with the independently supplied left
and right factors; it does not infer an inverse. It retains the intermediate
grades required for the requested final output. Handwritten two-product
compositions must do the same: selecting vectors immediately after $Rx$
would generally discard grades needed by the second product.

Repeated floating-point multiplication can accumulate a deviation from
$R\widetilde R=1$. Monitor this relation when storing a long composition.
If only the final transformed values are needed, composing induced actions
also avoids storing a rotor for every step.
