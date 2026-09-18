# Bivector Exponentials and Actions

The exponential of a bivector $B$ is the even multivector

\[
\exp(B) = \sum_{k=0}^{\infty}\frac{B^k}{k!}.
\]

Its occupied grades and evaluation cost depend on the dimension, signature,
and algebraic structure of $B$. Planning selects an executor from the algebra,
layouts, dtype, and device.

## Finite closures

### Simple closure

When $B^2=s$ is scalar, every power of $B$ lies in the span of $1$ and $B$:

\[
\exp(B)=C(s)+S(s)B,
\]

where

\[
C(s)=
\begin{cases}
\cos(\sqrt{-s}) & s<0,\\
1 & s=0,\\
\cosh(\sqrt{s}) & s>0,
\end{cases}
\qquad
S(s)=
\begin{cases}
\dfrac{\sin(\sqrt{-s})}{\sqrt{-s}} & s<0,\\
1 & s=0,\\
\dfrac{\sinh(\sqrt{s})}{\sqrt{s}} & s>0.
\end{cases}
\]

This covers all bivectors for $n\leq3$.

### Biquadratic closure

For $4\leq n\leq5$, write

\[
B^2=s+K,
\]

where $s$ is scalar and $K$ has grade 4. In these dimensions $K^2$ is scalar,
so the exponential closes over

\[
1,\quad B,\quad K,\quad BK.
\]

The closed formula evaluates the four scalar coefficients from the two roots
of the resulting quadratic relation. Real and complex root pairs use the same
finite closure with different scalar coefficient formulas.

## Materializing the exponential

`algebra.bivector_exp(B, input=bivectors)` returns coefficients of $\exp(B)$
exactly as formulated. The convention $\exp(-B/2)$ must be applied explicitly. Compact grade-2 input
defaults to an even-grade output layout. A full-basis input is first projected
to grade 2.

An explicit output layout selects coefficients of the exponential; it does
not redefine multiplication inside the power series. In particular,
repeatedly projecting each power to a narrow output would generally give
a different result.

Beyond finite closures, built-in methods use Taylor evaluation with scaling
and squaring or exponentiate the left-multiplication operator on the even
subalgebra. Materialized built-in exponentials support
dimensions 2 through 12, subject to resource limits. The Taylor path requires
coefficient L1 norm at most 65,536.

Intermediate computations may span the full even subalgebra even when the input or requested output uses a compact layout. The full even subalgebra has $2^{n-1}$ lanes.

## General numerical evaluation

For a submultiplicative coefficient L1 norm \(\beta=\sum_i|b_i|\), the
Taylor remainder after degree \(m\) obeys

\[
\left\|\exp(B)-\sum_{j=0}^m\frac{B^j}{j!}\right\|_1
\le e^\beta\frac{\beta^{m+1}}{(m+1)!}.
\]

The normalized basis metric factors have magnitude at most one, so this
coefficient norm bounds the product as required. The estimate describes
polynomial truncation before floating-point roundoff.

The Taylor implementation scales \(B\) until its L1 norm is at most one,
evaluates a polynomial, and squares back:

\[
\exp(B)=\left[\exp(B/2^s)\right]^{2^s}.
\]

It uses degree 12 for float32 and degree 18 for float64, with at most 16
squarings under the stated input envelope. The small-input path can evaluate
only grades needed by the requested output. Scaling and squaring can require
the full even representation, because intermediate grades contribute to later
products. Squaring propagates both polynomial error and roundoff; the simple
remainder estimate is not an error bound for the complete computed result.

Alternatively, let \(L_B\) represent left multiplication by \(B\) on the even
subalgebra and let \(e_0\) be its scalar-identity coordinate vector. Then

\[
\exp(B)=\exp(L_B)e_0.
\]

Here \(e_0\) denotes the scalar coordinate, not a Clifford generating vector.
The operator has \(2^{n-1}\) rows and columns. Its matrix exponential provides
a reference construction but carries the corresponding storage and interaction
cost. The current materialized matrix route executes on CPU and returns
coefficients to the input device through differentiable transfers.

## Removable limits and conditioning

The finite closures contain expressions such as
\(\sin(\sqrt{-s})/\sqrt{-s}\), whose limit at zero is one. Evaluating this as a
literal quotient loses both the value and useful gradients at zero.
Biquadratic coefficients also contain divided differences that subtract nearly
equal values near repeated roots.

The implementation uses polynomial expansions near these removable limits,
with thresholds derived from dtype epsilon and the omitted coefficient terms.
For example, balancing a first omitted term \(x^m/D\) against roundoff
amplification \(u/x^k\) gives a transition scale

\[
x_{\mathrm{cut}}=(uD)^{1/(m+k)},\qquad
u=\operatorname{finfo}(\mathrm{dtype}).\mathrm{eps}.
\]

This explains why lower precision calls for a wider polynomial interval.
The thresholds stabilize scalar coefficient evaluation without changing
the Clifford exponential into a low-rank approximation.

Large hyperbolic generators can produce large exponential coefficients.
Finite closure bounds the algebraic expansion but not coefficient magnitude: a finite input can still exceed the representable output range.
General Taylor and matrix routes require float32 or float64; on MPS those routes are limited to float32. Closed-form routes have separate dtype constraints.

## Applying the induced action

A grade-2 versor action uses the convention

\[
R=\exp(-B/2),\qquad x'=Rx\widetilde R.
\]

For a vector, define $G(B)x=-\tfrac12(Bx-xB)$. The same action is

\[
x'=\exp(G(B))x.
\]

Here $G(B)$ is an $n\times n$ matrix. Exponentiating it and lifting its
action to the requested grades avoids materializing all coefficients of $R$.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))
bivectors = algebra.layout((2,))
action = algebra.plan_versor_action(
    grade=2, input=vectors, parameter=bivectors, output=vectors,
)
x = torch.tensor([1.0, 0.0, 0.0])
B = torch.tensor([torch.pi / 2, 0.0, 0.0])
rotated = action(x, B)
assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
```

If \(M=\exp(G(B))\), its grade-\(k\) action is the exterior power
\(\bigwedge^k M\): apply \(M\) to each vector factor and take their wedge.
In a basis, its coefficients are \(k\times k\) minors of \(M\).
This explains grade preservation and why compact actions can avoid the
exponentially large rotor representation. High-grade lifts still have
combinatorial cost.

For the vector metric matrix \(\eta=\operatorname{diag}(1,\ldots,-1,\ldots,0,\ldots)\),
the induced generator satisfies \(G^\mathsf{T}\eta+\eta G=0\), hence
\(M^\mathsf{T}\eta M=\eta\) in exact arithmetic. With an indefinite or
degenerate signature this does not mean preserving Euclidean coefficient
length. The exponential matrix remains invertible, including for a
degenerate metric.

The action preserves grades; an explicit output declaration selects which
grades to retain. Use `bivector_exp` when the exponential's coefficients are
needed and `versor_action` when applying the generated transformation.

PyTorch differentiates the executed numerical map. Floating-point error,
large generator magnitudes, and backend support affect numerical behavior
even when the mathematical operation is fixed.
