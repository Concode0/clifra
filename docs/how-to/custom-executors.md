# Provide a custom executor

An executor provider implements a mathematical operation on compact coefficient
tensors. It receives public tensor contracts, requested dtype and device, and
operation names. It does not receive an algebra, planner, or built-in algorithm
configuration. A returned `nn.Module` must preserve the requested mathematics,
ordinary broadcasting, and differentiation through its tensor computation.

This provider implements only scalar geometric products. Everything else is
explicitly rejected so the registry can try another provider.

```python
from dataclasses import dataclass
import torch
from torch import nn
from clifra import ExecutorRegistry, make_algebra
from clifra.core.executors import Assessment, ExecutorRequest, Rejected

@dataclass(frozen=True)
class Preparation:
    dtype: torch.dtype
    device: torch.device

class ScalarProduct(nn.Module):
    def __init__(self, preparation):
        super().__init__()
        self.register_buffer("one", torch.ones(
            (), dtype=preparation.dtype, device=preparation.device
        ))

    def forward(self, left, right):
        return left * right * self.one

@dataclass(frozen=True)
class ScalarProvider:
    identity: tuple[str, str] = ("product", "example_scalar")

    def assess(self, request: ExecutorRequest):
        if request.family != "product" or request.operation != "geometric_product":
            return Rejected("only geometric products are supported")
        contracts = (*request.inputs, request.output)
        if len(request.inputs) != 2 or any(
            contract is None or contract.layout.grades != (0,)
            for contract in contracts
        ):
            return Rejected("only scalar contracts are supported")
        return Assessment(
            lanes=1, pairs=1,
            preparation=Preparation(request.dtype, request.device),
        )

    def build(self, request, assessment):
        return ScalarProduct(assessment.preparation)

registry = ExecutorRegistry((ScalarProvider(), *ExecutorRegistry.default().providers))
algebra = make_algebra(3, dtype=torch.float64, registry=registry)
scalar = algebra.layout((0,))
product = algebra.plan_product(left=scalar, right=scalar, output=scalar)
x = torch.tensor([[2.], [3.]], dtype=torch.float64, requires_grad=True)
y = torch.tensor([4.], dtype=torch.float64, requires_grad=True)
result = product(x, y)
torch.testing.assert_close(result, x * y)
dx, dy = torch.autograd.grad(result.sum(), (x, y))
torch.testing.assert_close(dx, y.expand_as(x))
torch.testing.assert_close(dy, x.sum(dim=0))
assert any(isinstance(module, ScalarProduct) for module in product.modules())
```

Assessment must not allocate execution buffers. Its `lanes` and `pairs` are
conservative maxima for coefficient width and static pair/interaction footprint,
including coexisting child plans and larger known fixed-shape temporaries.
Caller-controlled batch dimensions are excluded. Zero means no additional
requirement; the input and output widths are guarded independently. A successful
assessment accepts the declared operation without deliberate approximation or
truncation. Ordinary floating-point roundoff remains possible.

Preparation is provider-owned metadata. Keep it immutable, allocate tensors in
`build`, and use the assessment that was supplied rather than reassessing.
The exact request and assessment objects are passed unchanged to `build`.
Provider identity and behavior must remain fixed for the lifetime of a registry;
the frozen provider above enforces that for its own configuration.

## Keep fallback and storage behavior

Registries are explicit and immutable, with unique `(family, route)` identities.
Supplying a registry replaces the default collection. Prepending an external
provider gives it precedence when eligible; appending it makes it a fallback
after built-in capability. Rejection or an over-budget assessment falls through
without calling that provider's `build`. An empty registry remains empty.

The wrapper converts canonical storage to compact storage and supplies pairwise
expansion. The provider still needs only `forward(left, right)`:

```python
from clifra import TensorContract

canonical = TensorContract.canonical(scalar)
full_product = algebra.plan_product(
    left=canonical, right=canonical, output=canonical
)
torch.testing.assert_close(
    full_product(scalar.full(x), scalar.full(y)), scalar.full(x * y)
)
pairwise = algebra.plan_product(
    left=scalar, right=scalar, output=scalar, pairwise=True
)
items = torch.tensor([[4.], [5.], [6.]], dtype=torch.float64)
torch.testing.assert_close(pairwise(x, items), x[:, None] * items[None, :])

vector = algebra.layout((1,))
v = torch.tensor([1., 2., 3.], dtype=torch.float64)
torch.testing.assert_close(
    algebra.geometric_product(v, v, left=vector, right=vector, output=scalar),
    v.square().sum().reshape(1),
)
```

The last operation falls back to a built-in provider because its inputs are
vectors. No global registration changes another algebra. Register tensor state
as parameters or buffers so normal module movement works; test forward values,
gradients, broadcasting, storage conversion, and any intended compilation
backend for the supported contracts.
