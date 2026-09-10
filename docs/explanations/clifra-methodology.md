# Geometric Representations and Parameters

A layout defines the coefficient space. The same tensor representation can be
used as fixed data, a differentiable input, or a PyTorch parameter. Its grade
does not determine whether it is a model weight, a sample, or an operand.

The operation gives the coefficients their role. A vector may enter a product,
define a reflection normal, or be projected onto a blade. A mixed-grade value
can be added, reversed, or multiplied without interpreting it as a geometric
transformation.

## Vectors, blades, and mixed grades

A grade-1 layout contains the \(n\) vector coefficients. Its signed square
comes from the signature; its coefficient length comes from ordinary Euclidean
coordinates. Choosing one as a normalization rule does not make it the other.

A simple grade-\(k\) blade \(A=a_1\wedge\cdots\wedge a_k\) represents an oriented
subspace with scale. Constructing it from vector factors ensures
decomposability, although dependent factors can produce zero. Storing arbitrary
grade-\(k\) coefficients is more general: a homogeneous multivector need not be
a simple blade. A layout declaration alone cannot establish the assumptions
required for a blade inverse or subspace projection.

A mixed-grade layout is a direct sum of coefficient spaces. For instance,
grades 0 and 1 store a scalar and vector in one tensor. Reversal, projection,
and geometric multiplication each give this value a well-defined meaning
without additional parameter constraints.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))
scalars = algebra.layout((0,))
mixed = algebra.layout((0, 1))
v = torch.tensor([1.0, 2.0, 0.0])
a = mixed.convert(torch.tensor([2.0]), scalars) + mixed.convert(v, vectors)
assert a.shape == (4,)
square = algebra.geometric_product(
    a, a, left=mixed, right=mixed, output=mixed,
)
torch.testing.assert_close(square, torch.tensor([9.0, 4.0, 8.0, 0.0]))
```

Here \((2+v)^2=4+4v+v^2=9+4v\). The output selection retains the scalar and
vector terms. This calculation neither needs a full multivector tensor nor
introduces a transformation parameter.

## Bivectors and explicit rotors

A bivector is a grade-2 value. It can represent an oriented area, enter a
commutator, or generate an action. In dimensions four and above, a general
bivector can contain several independent plane components; it need not be a
single simple blade.

Exponentiation is one specialized use. For a bivector \(B\),

\[
R=\exp(-B/2),\qquad \widetilde R=\exp(B/2),\qquad R\widetilde R=1.
\]

The identity follows from \(\widetilde B=-B\) and the commuting exponential
factors. Generator coordinates are useful when varying an action smoothly.
Explicit even-grade rotor coefficients are useful when composing, comparing,
or exporting group elements. An arbitrary even multivector is not necessarily
a rotor.

These representations are not interchangeable coordinate systems. The action
identifies \(R\) with \(-R\), generator coordinates can be periodic, and a
product of rotors need not have a unique bivector logarithm. Adding generators
implements composition only when they commute. See
[Bivector Exponentials and Actions](bivector-exponential.md) for the operation
and its numerical evaluation.

## Parameters and constraints

Any declared coefficient tensor can be a PyTorch parameter. A shared product
operand of shape `[lanes]` broadcasts across samples; separate operands of
shape `[samples, lanes]` receive separate gradients. Parameter ownership adds
no Clifford semantics beyond the declaration and the operations using it.

Constraints follow the represented object. Ordinary mixed-grade coefficients
may need no adjustment. A blade built from vector factors preserves
decomposability under changes to those factors, but can become singular.
A reflection normal must stay non-null for the exact reflection.
An additive update to a stored rotor need not preserve rotor membership,
whereas a bivector always defines its mathematical exponential.

A coefficient penalty constrains a representation. A signed invariant,
projection residual, or transformed-value loss constrains a calculation.
Choose these explicitly rather than treating every coefficient tensor as a
generator or every grade as a manifold.

## Composition and invariants

A sum or product combines represented values; an output layout selects grades.
An intermediate projection can remove terms needed later, so a narrow result
cannot always replace the complete value in another expression.

An exact reflection or rotor action preserves the signature-sensitive vector
form. A subsequent coordinate gate or generic tensor map need not preserve it.
Likewise, a coefficient normalization changes scale but does not certify blade
simplicity or a signed invariant. Mathematical claims should refer to the
complete calculation being used.
