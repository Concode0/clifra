# Bivector Coordinate Fields

This advanced application combines sampled generators with planned actions.
It is independent of the core representation and execution contracts.

A field of bivectors assigns a local generator to each sample identity.
Applying a planned action to those generators and vector values produces a
field of local transformations. Sampling, interpolation, and composition use
ordinary PyTorch operations around the Clifford action.

The distinction between a value and its sampling identity matters. A material
coordinate, grid coordinate, or persistent point label can select a generator
while the vector being transformed changes.

## From a bivector to a local action

Hold a bivector \(B\) fixed and define

\[
R(t)=\exp(-tB/2),\qquad
x(t)=R(t)x(0)\widetilde{R(t)}.
\]

Bivector reversal gives \(\widetilde B=-B\), hence
\(\widetilde{R(t)}=\exp(tB/2)\). Differentiation gives

\[
\frac{dx(t)}{dt}
=-\frac12\bigl(Bx(t)-x(t)B\bigr)=G(B)x(t).
\]

The commutator with a bivector maps vectors to vectors, so \(G(B)\) is an
ordinary \(n\times n\) matrix on vector coefficients. Since the generator is
constant during this local action,

\[
x(1)=\exp(G(B))x(0).
\]

The planned grade-2 action evaluates this map without first materializing the
even-grade coefficients of \(R\). This construction works in positive,
indefinite, and degenerate signatures; the signature determines \(G(B)\).

## Sampling and ordered composition

Let \(\xi_i\) identify sample \(i\), and let \(B_s^\Theta(\xi_i)\) be its
generator at stage \(s\). A field of \(S\) ordered actions is

\[
\phi_\Theta(x_i,\xi_i)=
\exp(G(B_{S-1}^\Theta(\xi_i)))\cdots
\exp(G(B_0^\Theta(\xi_i)))x_i.
\]

The sample index is a leading tensor axis. Stage order belongs to the
composition, not to broadcasting. The number of stages need not represent
physical time.

A fixed linear map can lift sampled coordinates \(z_s^\Theta(\xi)\) into
bivector coefficients, \(B_s^\Theta(\xi)=M z_s^\Theta(\xi)\). Interpolation
or a coordinate function can supply \(z\). These choices control variation of
the generators; they do not interpolate the values being transformed.

In general,

\[
\exp(G(B_2))\exp(G(B_1))\ne
\exp(G(B_1))\exp(G(B_2)).
\]

Reordering stages changes the result. Replacing the composition with the
exponential of a sum is justified when the generators commute, not merely
because their coefficients share the same layout.

## Differentiation through the field

For a sampling function differentiable in \(\Theta\), gradients pass through
sampling, the optional linear lift, each matrix exponential, and the ordered
actions. A loss on transformed coordinates therefore differentiates back to
the field coefficients.

A field sampled on one grid can be evaluated on another if the sampling
function is defined at the new identities. This is evaluation of the same
function, not a guarantee about interpolation accuracy or an arbitrary
resampling rule. Sparse observations constrain the chosen field family; they
do not uniquely determine all possible deformations between observations.

## Indexed inversion and global geometry

For fixed \(\xi\), each action is invertible:

\[
\exp(G(B_s(\xi)))^{-1}=\exp(-G(B_s(\xi))).
\]

The inverse composition applies the same generators with opposite signs in
reverse stage order. It must retain the original sampling identity. Sampling
again at transformed coordinates can choose a different generator and break
this inverse relation.

Local invertibility does not establish global injectivity of an assembled
field. Different identities can select different invertible actions whose
outputs coincide. In the coordinate-driven case \(f(x)=A(x)x\),

\[
Df(x)[h]=A(x)h+\bigl(DA(x)[h]\bigr)x.
\]

The second term is absent from the fixed-identity action. Thus a locally
metric-preserving matrix \(A(x)\) need not make the spatial map an isometry,
or even make its Jacobian nonsingular everywhere. Global properties need
conditions on how the generators vary over the domain.