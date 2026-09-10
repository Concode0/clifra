# Reflect vectors and multivectors

For a non-null vector normal $n$, reflection of a vector $x$ is
$x-2\langle xn\rangle_0 n/n^2$. For a multivector $X$, the induced action is
$n\widehat X n^{-1}$, where the hat denotes grade involution. This leaves
scalars fixed and extends the vector reflection to exterior products.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, dtype=torch.float64)
vectors = algebra.layout((1,))
x = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], dtype=torch.float64)
normal = torch.tensor([1., 0., 0.], dtype=x.dtype)
reflection = algebra.plan_strict_reflect(input=vectors, normal=vectors)
result = reflection(x, normal)
expected = x * torch.tensor([-1., 1., 1.], dtype=x.dtype)
torch.testing.assert_close(result, expected)
torch.testing.assert_close(reflection(result, normal), x)
torch.testing.assert_close(reflection(x, 3 * normal), result)
```

For mixed grades, reflect each basis blade through its constituent vectors.
With normal $e_1$, a blade changes sign exactly when it contains $e_1$.
The scalar and the $e_{23}$ plane remain fixed.

```python
mixed = algebra.layout((0, 1, 2))
# Lanes: 1, e1, e2, e12, e3, e13, e23.
X = torch.tensor([1., 2., 3., 4., 5., 6., 7.], dtype=x.dtype)
reflect_mixed = algebra.plan_strict_reflect(input=mixed, normal=vectors)
reflected = reflect_mixed(X, normal)
signs = torch.tensor([1., -1., 1., -1., 1., -1., 1.], dtype=X.dtype)
torch.testing.assert_close(reflected, X * signs)
torch.testing.assert_close(reflect_mixed(reflected, normal), X)
```

`strict_reflect` and `plan_strict_reflect` use the exact computed denominator
and reject a zero value. In indefinite signatures a nonzero normal can be
null; no hyperplane reflection of this form exists there. An arbitrarily
small nonzero denominator is accepted by the strict variant and can amplify
roundoff and gradients.

`reflect` and `plan_reflect` clamp the magnitude of the inverse denominator
at the algebra's squared machine epsilon, retaining a negative sign and
using a positive sign for zero. This supplies a numerical fallback, not a
reflection in a null hyperplane. For example, a zero normal produces zero
through this composition, so involutivity does not hold at that fallback.

`plan_versor_action(grade=1, input=..., parameter=vectors)` also represents
reflection away from singular inputs. Its action routes normalize and
regularize normals differently; do not rely on identical near-null
fallback values between it and `reflect`. Use a strict reflection when
undefined normals must be reported rather than regularized.
