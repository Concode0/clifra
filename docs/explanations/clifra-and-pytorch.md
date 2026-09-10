# Tensor Composition and PyTorch Ownership

clifra operations consume and return ordinary PyTorch tensors. A layout
defines the coefficient space; a tensor contract also specifies its physical
storage. These declarations live on call sites or planned operations, not on a
special tensor subclass.

The final axis holds Clifford coefficients. Other axes can represent samples,
channels, quadrature points, or any application-defined dimensions. Their
names have no effect on execution: ordinary shape and broadcasting rules
determine how tensors combine.

## Composition preserves declarations

A scalar multiplication, sum over a sample axis, or reshaping of leading
dimensions leaves coefficient meanings intact. Addition requires aligned
layouts. An elementwise gate changes coefficients within a layout, but is not
generally a Clifford multiplication or a metric-preserving transformation.

For layouts \(L_1\) and \(L_2\), convert into a common layout before adding.
Zero-filled missing lanes represent absent components. Conversion back to a
smaller layout is a projection, so a round trip only preserves components in
the intersection.

The same principle applies to a chain of products. An output projection in
the middle of a chain can remove components needed later. Associativity of
the geometric product does not imply
\(\pi(\pi(AB)C)=\pi(A(BC))\) for an arbitrary intermediate projection \(\pi\).
Declare intermediate grades from the mathematical calculation.

## Autograd differentiates coefficient values

A planned operation contains fixed algebraic structure and a differentiable
tensor program. PyTorch differentiates with respect to its tensor inputs and
any parameters feeding them. Signatures, grade choices, and layout indices are
discrete declarations, not differentiable variables.

Broadcasting also determines gradient accumulation. If one product operand is
shared across many samples, its gradient sums contributions from those uses.
Giving each sample its own operand changes both the parameterization and the
gradient shape; no separate clifra routing mechanism is required.

Coefficient-space losses can use ordinary PyTorch arithmetic. For example,
`(prediction - target).square().mean()` compares aligned coefficients. A signed
Clifford form is a different objective, discussed in
[Signatures and Algebraic Behavior](signatures-and-learning.md).

## Retaining a computation in a module

An ordinary `nn.Module` can register a `PlannedOperation` as a child. Use
`CliffordModule` when the owner also needs an algebra reference whose placement
defaults follow module movement.

```python
import torch
from clifra import CliffordModule, make_algebra


class ScalarForm(CliffordModule):
    def __init__(self, algebra):
        super().__init__(algebra)
        vectors = algebra.layout((1,))
        self.direction = torch.nn.Parameter(
            torch.ones(vectors.dim, device=algebra.device, dtype=algebra.dtype)
        )
        self.product = algebra.plan_product(
            left=vectors, right=vectors, output=algebra.layout((0,)),
        )

    def forward(self, values):
        return self.product(values, self.direction)


form = ScalarForm(make_algebra(2, 1))
values = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
torch.testing.assert_close(form(values), torch.tensor([[1.0], [-1.0]]))
form(values).sum().backward()
torch.testing.assert_close(form.direction.grad, torch.tensor([1.0, 0.0, -1.0]))
```

The parameter stores three vector coefficients. The planned product returns
\(\langle xw\rangle_0\) in \(Cl(2,1)\), retaining the signature's negative
third direction. The shape `[3]` shares this operand over leading input
dimensions. The same operation can be used with fixed tensors without a
parameter or owning module.

## Movement and persistence

A plan's `.to(device=..., dtype=...)` moves its owned tensor state.
`AlgebraContext.to(...)` instead changes defaults for future plans. Existing
input tensors still need the appropriate placement.

When a `CliffordModule` moves, it obtains an algebra context with the new
defaults. Other owners of the original context keep their defaults.
An ordinary `nn.Module` moves registered parameters, buffers, and child plans,
but does not know how to update a plain algebra reference.

Reconstruct plans from signature, layouts, storage forms, and operation
arguments before loading application parameters. Do not treat a saved private
kernel layout as an interchange format. Post-update callbacks and custom
executor providers are also application configuration, not learned tensor
state.

## Compilation and numerical behavior

Construct fixed plans before entering a compiled function. The retained module
executes without Python-side planning, while the compiler handles its tensor
program. Some operations include runtime assertions or conditional numerical
work; not every backend lowers every supported eager operation.

Graph capture and backend code generation are separate checks. A
`torch.compile(..., backend="eager", fullgraph=True)` comparison checks capture;
it does not establish performance or compatibility with another backend.
Check forward values and gradients on the intended dtype and device.

Autograd follows the implemented formulas, including clamps and piecewise
numerical choices. A strict inverse can be ill-conditioned near a null input;
a norm has a nonsmooth point at zero; a large exponential can overflow.
Differentiability through the tensor program does not remove these properties
of the computation. See [Compile Clifford computations](../how-to/compilation.md)
for a reproducible capture and gradient check.
