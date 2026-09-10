# Signatures and Algebraic Behavior

The signature of $Cl(p, q, r)$ defines the square of each generating basis
vector:

\[
e_i^2 =
\begin{cases}
+1 & 1 \le i \le p,\\
-1 & p < i \le p + q,\\
0 & p + q < i \le p + q + r.
\end{cases}
\]

These signs propagate through every Clifford product. They are separate from
the positive coefficient geometry that clifra also exposes for losses,
diagnostics, and stable optimization.

## Signed forms and lane energy answer different questions

For compact coefficients $x_i$, lane energy is

\[
E_{\mathrm{lane}}(x) = \sum_i x_i^2.
\]

It is positive definite and independent of metric signs. It measures the size
of a coefficient tensor in its declared layout. `lane_energy`, `lane_norm`,
`lane_distance`, and the per-grade lane functions use this geometry.

A Clifford form is built from algebraic products. `signature_norm_squared`
computes

\[
Q(x) = \langle x\widetilde{x}\rangle_0,
\]

with signs determined by the signature and blade grades. Depending on the form
and convention, its value can be positive, negative, or zero for a nonzero
multivector. Null directions in a degenerate signature make the last case
fundamental rather than exceptional.

| Quantity | Signature-sensitive | Positive definite | Typical use |
| --- | --- | --- | --- |
| Lane energy | No | Yes | Coefficient scale, regularization, stable distances |
| Per-grade lane energy | No | On the selected grade | Grade distribution and diagnostics |
| Signature or conjugate scalar form | Yes | No in general | Algebraic invariants and metric-aware constraints |
| Magnitude derived with `abs` | Yes, before `abs` | Nonnegative, but not a norm in every signature | Stable scale based on a signed form |

Calling both quantities a norm without qualification hides an important design
choice. A signed form represents the algebraic metric. Lane energy represents a
Euclidean geometry on the coordinate lanes used by the optimizer.

## Signature determines product behavior

A planned product stores fixed left-lane positions, right-lane positions,
output positions, and coefficients. The coefficients include permutation signs,
metric signs, and zero products from null basis directions. Planning removes
interactions that are known to be zero and execution performs the remaining
gather, multiply, and reduction operations.

The resulting tensor program is signature-specific even though its runtime
operations look like ordinary PyTorch arithmetic. Changing $Cl(3, 0, 0)$ to
$Cl(1, 2, 0)$ changes more than coefficient names: it changes basis products,
signed forms, exponential behavior, and potentially executor eligibility.

## Forward and backward use the same algebra

Autograd differentiates the actual tensor program used in the forward pass. For
an exact planned product with fixed coefficients,

\[
y_k = \sum_{i,j} c_{ijk} a_i b_j,
\]

the backward pass contains the same signature-dependent coefficients
$c_{ijk}$:

\[
\frac{\partial L}{\partial a_i}
= \sum_{j,k} c_{ijk} b_j \frac{\partial L}{\partial y_k}.
\]

Backpropagation retains the algebraic signs from the forward calculation; the
gradient follows that signed tensor program, with no substituted positive
metric.

Using lane energy as a loss or regularizer introduces a positive Euclidean
objective on coefficient space. That choice changes the optimization geometry
and therefore the path taken by learning, while leaving the model's Clifford
product unchanged. An indefinite algebra can therefore use a positive
coefficient-space objective and still retain its original signature.

Use lane quantities for coefficient scale and distances. Use a signed form
for a signature-sensitive invariant or constraint. Taking `abs` of a signed
form yields a nonnegative quantity but does not make it a positive-definite
norm.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(1, 1)
vectors = algebra.layout((1,))
x = torch.tensor([1.0, 1.0])
assert algebra.lane_energy(x, input=vectors).item() == 2.0
assert algebra.signature_norm_squared(x, input=vectors).item() == 0.0
```

This nonzero vector is null for the signed form. Its coefficient energy
remains positive.

## Scalar forms and output axes

`scalar_product(A, B)` computes $\langle AB\rangle_0$.
`conjugate_scalar_form(A, B)` computes $\langle\overline A B\rangle_0$,
where the bar is Clifford conjugation, not complex conjugation. Reversal,
conjugation, and no involution give different signs. Even in a Euclidean
signature, the scalar part of a bivector square is negative while its
coefficient energy is positive.

Lane dot products and distances align layouts by blade identity before
comparing coefficients. Lane energies and scalar forms preserve all leading
axes and return a final singleton axis. Per-grade energy returns $n+1$
entries indexed by grade, including zeros for grades absent from the layout.
That final axis is a list of grade measurements, not a multivector layout.

For zero input, every grade energy is zero. `lane_grade_distribution` divides
by total energy plus `eps`, so its entries sum to slightly less than one for
nonzero data and to zero for all-zero data. It is a stabilized coefficient
summary, not a probability distribution inferred from the algebra.

## Numerical behavior at zero and near null values

`lane_energy` is a polynomial in real coefficients. `lane_norm` takes its
square root, which has a nonsmooth norm point at the zero vector. No epsilon
is added by that helper. If a differentiable objective needs smooth behavior
there, squared coefficient energy often expresses the intended penalty
without a square root.

Signed cancellation is a separate issue. Large coefficients can produce a
small signed form in an indefinite algebra, making relative error large.
Inversion divides by such a form and can magnify that error. Strict geometry
rejects an exactly zero denominator; it does not set a conditioning threshold.
Stabilized geometry changes the denominator near zero, and autograd follows
that changed numerical expression.

Compare quantities in the precision and range used by the application.
Coefficient energy can reveal large values hidden by signed cancellation;
the signed form still determines the algebraic invariant being checked.
