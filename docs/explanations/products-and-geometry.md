# Products, Grades, and Blade Geometry

For orthogonal basis vectors, the Clifford relation is

\[
e_i e_j+e_j e_i=2g_i\delta_{ij},\qquad g_i\in\{1,-1,0\}.
\]

Products of ordered basis blades follow by swapping generators and applying
their squares. A zero metric factor removes a term. Linearity then defines
the product of coefficient tensors.

## Product conventions

For homogeneous \(A_r\) and \(B_s\), clifra uses:

| Operation | Definition |
| --- | --- |
| Geometric product | \(A_rB_s\) |
| Exterior product | \(\langle A_rB_s\rangle_{r+s}\) |
| Left contraction, \(r\le s\) | \(\langle A_rB_s\rangle_{s-r}\) |
| Right contraction, \(r\ge s\) | \(\langle A_rB_s\rangle_{r-s}\) |
| Symmetric product | \((AB+BA)/2\) |
| Commutator product | \(AB-BA\) |
| Anticommutator product | \(AB+BA\) |

Contractions are zero when their grade inequality fails. Mixed-grade inputs
extend these definitions by summing homogeneous pairs. There is no implicit
reversal in the contractions and no factor of one half in
`commutator_product`. These conventions matter when translating formulas from
another geometric algebra library.

The possible geometric-product grades lie between \(|r-s|\) and
\(\min(r+s,2n-r-s)\) in steps of two. Signature degeneracy can make additional
coefficients zero. A declared output retains only the requested grades.

For vectors, \(ab=\langle ab\rangle_0+a\wedge b\). In higher grades, a scalar
product is not generally a Euclidean coefficient dot product.

## Unary operations

For grade \(k\), the unary signs are

\[
\widetilde{A_k}=(-1)^{k(k-1)/2}A_k,\qquad
\widehat{A_k}=(-1)^k A_k,\qquad
\overline{A_k}=(-1)^{k(k+1)/2}A_k.
\]

They correspond to `reverse`, `grade_involution`, and
`clifford_conjugation`. Reversal reverses factor order,
\(\widetilde{AB}=\widetilde B\widetilde A\); grade involution preserves it.
`grade_projection` selects grades through an explicit output declaration.

These operations preserve leading axes. They can change lane signs or return
a narrower layout without canonical materialization.

## Inverting a blade

For a non-null blade \(A\),

\[
A^{-1}=\frac{\widetilde A}{\langle A\widetilde A\rangle_0}.
\]

The blade assumption matters. For a general multivector,
\(A\widetilde A\) need not be scalar, so dividing by its scalar part is not a
general multivector inverse. A homogeneous grade declaration alone does not
prove that a value is a simple blade.

The strict variant uses the computed denominator and raises if it is exactly
zero. It does not certify simplicity, finite values, or good conditioning.
The stabilized variant clamps the denominator magnitude at a positive
epsilon while preserving its sign, taking a zero denominator as positive.
Near zero, the resulting value no longer satisfies the exact inverse identity.

## Projection and rejection

For a vector \(x\) and an invertible blade \(A\) representing a subspace,

\[
\operatorname{proj}_A(x)=(x\mathbin{\lrcorner}A)A^{-1},
\qquad
\operatorname{rej}_A(x)=x-\operatorname{proj}_A(x).
\]

The same API evaluates the declared contraction/product expression for
multivectors; its interpretation should follow that expression. In an
indefinite metric, projection onto a null subspace is not made well-defined
by clamping an inverse.

Strict projection and rejection use the strict inverse denominator.
Stabilized forms use the guarded inverse, so exact decomposition identities
involving the projected subspace can fail near the clamp. Rejection retains
the input layout to make subtraction well-defined.

## Reflections and sandwiches

For a non-null vector normal \(a\), reflection extends to multivectors as

\[
X'=a\widehat X a^{-1}.
\]

For a vector this is \(-axa^{-1}\); for a scalar it leaves the value unchanged.
Applying a minus sign uniformly to an entire multivector would give the wrong
parity. `reflect` and `strict_reflect` use this grade-involuted argument.

The grade-1 generated action is a vector reflection lifted grade by grade.
Away from singular denominators it agrees with the sandwich formula.
Its numerical normalization and guards differ from the direct inverse
composition, so their regularized null behavior need not agree.

A general sandwich `sandwich_product(L, X, R, ...)` computes \(LXR\).
It does not infer that \(R=L^{-1}\). For a rotor \(R\), the usual action is
\(RX\widetilde R\); this inverse relation follows from rotor membership,
not simply from an even-grade layout.

See the how-to guides for [products](../how-to/products.md),
[reflections](../how-to/reflections.md), and
[safe blade geometry](../how-to/blade-geometry.md).
