# Signatures and Algebraic Behavior

The signature of $Cl(p, q, r)$ defines the square of each generating basis
vector:

\[
e_i^2 =
\begin{cases}
+1 & i < p,\\
-1 & p \le i < p + q,\\
0 & p + q \le i < p + q + r.
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

A Clifford form is built from algebraic products. For example, a signature norm
squared is a scalar projection such as

\[
Q(x) = \langle \widetilde{x}x\rangle_0,
\]

with signs determined by the signature and blade grades. Depending on the form
and convention, its value can be positive, negative, or zero for a nonzero
multivector. Null directions in a degenerate signature make the last case
fundamental rather than exceptional.

| Quantity | Signature-sensitive | Positive definite | Typical use |
| --- | --- | --- | --- |
| Lane energy | No | Yes | Coefficient scale, regularization, stable distances |
| Per-grade lane energy | No | Yes | Grade distribution and diagnostics |
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
$Cl(1, 2, 0)$ does not merely rename the same coefficients; it changes basis
products, signed forms, exponential behavior, and potentially executor
eligibility.

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

Clifra does not replace the algebraic signs with a positive metric during
backpropagation. The signed forward calculation determines the gradient.

Using lane energy as a loss or regularizer introduces a positive Euclidean
objective on coefficient space. That choice changes the optimization geometry
and therefore the path taken by learning, but it does not alter the Clifford
product implemented in the model. An indefinite algebra can be optimized with a
positive coefficient-space objective without being redefined as a Euclidean
Clifford algebra.

## Algebra and optimization geometry

Three layers of claims must remain separate:

1. **Algebraic definition.** Exact product, reverse, projection, and exact
   exponential routes implement the declared Clifford operations up to ordinary
   floating-point error.
2. **Numerical method.** Clamps, null fallbacks, filtered eigengradients, and
   spectral truncation define how difficult cases are computed stably.
3. **Learning objective.** Lane losses, signed invariants, regularizers, and
   domain constraints define what the model is trained to prefer.

A positive coordinate-space loss does not alter the Clifford product used by
the forward program. An invariant form of the Clifford algebra and an
optimization geometry on its coefficients are different definitions and should
not be presented as equivalent.

The same distinction applies to numerical safeguards. Taking the absolute value
before a square root yields a stable magnitude-like scalar, but it does not turn
an indefinite form into a positive-definite norm. Falling back to Euclidean
normalization near a null vector avoids division by zero, but it does not prove
that the result has unit signed norm.

## Scope of approximate exponentials

For spectral-local bivector exponentiation, the backward pass is the derivative
of the implemented local computation. Near repeated eigenvalues, a filtered
eigendecomposition suppresses unstable inverse spectral gaps. If plane
truncation occurs, both forward and backward belong to the truncated model.

This remains a valid differentiable numerical method, but identities that
depend on the exact full exponential must be validated to the required tolerance
for that method. The algebraic foundation specifies the target operation; it
does not make every approximation exact.

## Choosing a quantity for learning

Use lane geometry when the intended statement concerns coefficient scale,
regularization, or comparison in a stable positive space. Use a signed Clifford
form when the intended statement concerns the metric, an algebraic invariant,
or a geometric constraint. In an indefinite or degenerate signature, decide how
negative and null cases should affect the application rather than applying
`abs` without interpreting it.

For a complete model, verify both levels independently:

- compare planned algebraic operations with an exact reference where feasible;
- test the model-level invariant or scientific constraint actually claimed;
- monitor lane-scale diagnostics for numerical conditioning;
- characterize approximation and gradient behavior in the operating regime.

The separation preserves the meaning of the declared signature while permitting
a positive coordinate-space objective.
