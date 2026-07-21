# Why bivector coordinate fields work

The continuum solver is Clifra's end-to-end research demonstration of
layout-directed geometric learning. It implements a differentiable field of
local Clifford actions parameterized by sampled bivector generators. Clifra
supplies the structured coordinate spaces and compiled transformation law;
user-defined differentiable objectives determine what the field learns.

The continuum construction extends a Clifford action into a generator field
over a domain. Sampling determines how its local generators vary, paths
determine how their actions compose, and differentiable objectives determine
the transformation the field learns. The included physics-informed deformation
system is one complete realization of this construction.

## From a bivector to an action

Let $B$ be a bivector in $Cl(p,q,r)$, let $n=p+q+r$, and hold $B$ fixed over
one action step. Define

\[
R(t)=\exp(-tB/2).
\]

Because bivector reversal gives $\widetilde{B}=-B$, the rotor action on a
grade-1 value is

\[
x(t)=R(t)x(0)\widetilde{R(t)}
    =\exp(-tB/2)x(0)\exp(tB/2).
\]

Differentiating gives

\[
\frac{d x(t)}{dt}
=-\frac12\left(Bx(t)-x(t)B\right)
=G(B)x(t).
\]

Here, $G(B)$ is the $n\times n$ linear generator induced by the commutator on
the grade-1 subspace. Therefore

\[
x(1)=\exp(G(B))x(0).
\]

This is why bivector coefficients can be learned directly: the default planned
action maps their linear coordinate space into invertible linear actions on
grade 1. For an ordered path $B_1,\ldots,B_S$,

\[
x_S=\exp(G(B_S))\cdots\exp(G(B_1))x_0.
\]

The inverse uses $-B_S,\ldots,-B_1$ in that order.

## From one action to a field

`CoordinateChart` embeds input coordinates into grade 1 and extracts the
result. Let $\Theta$ denote the stored generator parameters. A sampler evaluates
the step generator $B_s^\Theta(\xi_i)$ at persistent sample label $\xi_i$:

\[
\phi_\Theta(X_i,\xi_i)
=D\!\left[
\exp\!\left(G(B_S^\Theta(\xi_i))\right)\cdots
\exp\!\left(G(B_1^\Theta(\xi_i))\right)E(X_i)
\right].
\]

`CoordinateFieldInput` keeps transformed values separate from persistent
material labels. Regular-grid sampling associates generators by tensor index;
RBF sampling associates them by arbitrary coordinates. Interpolation changes
the generator field, never the input coordinate values.

Training is ordinary differentiation through the action:

\[
\min_\Theta\;
\mathcal{L}\!\left(\operatorname{State}_\Theta(X,\xi),Y\right)
+\sum_k\lambda_k
\mathcal{P}_k\!\left(\operatorname{State}_\Theta(X,\xi),\Theta\right).
\]

Here, $\operatorname{State}_\Theta$ contains the input identity, transformed
values, and sampled generator weights exposed by `ContinuumState`. A criterion
or policy may use only the parts it needs. The mechanics example supplies one
choice of $\mathcal{L}$ and $\mathcal{P}_k$. Dimension, signature, sampling
domain, and objective are configurable parts of the construction.

## Minimizing exponential work

The field's normal grade-1 action does not construct a full even multivector
with `bivector_exp`. Clifra builds $G(B)$ directly and evaluates the smaller
`torch.matrix_exp` on the $n\times n$ vector representation. The explicit
`rotor_path()` and `rotors_for_input()` helpers use `bivector_exp` when a caller
asks to inspect rotors.

The field also avoids exponentiating weights repeated only by broadcasting:

- a global generator is exponentiated once per path step;
- a regular-grid generator is exponentiated once per spatial site and shared
  across batch axes;
- a coordinate-dependent generator is exponentiated once per sampled generator
  row; numerically equal rows are not deduplicated.

Path steps are evaluated separately because, in general,

\[
\exp(G_2)\exp(G_1)\ne\exp(G_1+G_2)
\]

when $[G_2,G_1]\ne0$. Merging such steps would change the parameterization.
The implementation therefore preserves path composition and removes only
broadcast-redundant exponential evaluations.

## Local inversion and global geometry

For the default planned action, a path has an inverse when it is evaluated with
the same persistent sample labels: negate its generators and reverse their
order. Projective extraction additionally requires a nonzero homogeneous
coordinate. An injected action module must satisfy the same inverse identity if
it is to retain this property.

Global injectivity remains a property of the assembled field, not of each local
matrix exponential. Application constraints may encourage it; the validation
required by the application must establish the property actually claimed.

`research/continuum_solver/examples/bivector_field_basics.py` is the compact
introduction; it learns a local field and verifies its structural properties.
`research/continuum_solver/examples/physics_informed_deformation_design.py` is
the complete physics-informed application.
