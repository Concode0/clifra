# Planned Differentiation

A planned operation fixes the algebraic computation before coefficient values
arrive. This is useful when a product appears repeatedly in a training loop:
its signature, input and output layouts, and storage contracts stay the same
while parameters and batches change.

## Fix the operation

For Euclidean vectors, the scalar part of $xy$ is their inner product. We can
plan only that output grade instead of producing scalar and bivector lanes and
discarding the latter.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0, dtype=torch.float64)
vectors = algebra.layout((1,))
scalars = algebra.layout((0,))
inner = algebra.plan_product(
    op="geometric_product", left=vectors, right=vectors, output=scalars,
)
```

The plan resolves the requested basis interactions and selects an executor.
Its callable interface consumes tensors satisfying `inner.inputs` and returns
a tensor satisfying `inner.output`. Planning fixes no coefficient values,
sample count, or autograd graph. The operation is a PyTorch module with its own
device and dtype placement; this example creates both it and its inputs in
double precision.

## Differentiate through repeated calls

```python
direction = torch.nn.Parameter(torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64))
samples = torch.tensor(
    [[1.0, 2.0, 0.0], [-1.0, 0.0, 3.0]], dtype=torch.float64,
)
scores = inner(samples, direction)
assert scores.shape == (2, 1)
loss = scores.square().mean()
loss.backward()

expected_gradient = 2 * (scores.detach() * samples).mean(dim=0)
torch.testing.assert_close(direction.grad, expected_gradient)
```

The scalar layout still has a final coefficient axis of width one. The loss
reduces both sample and scalar axes. Broadcasting shares the direction across
samples, and autograd adds their contributions to its gradient.

The gradient above is a derivative in coefficient coordinates. In $Cl(3,0)$
the scalar product also agrees with the ordinary dot product. This equality
does not extend unchanged to an indefinite signature: metric signs enter the
Clifford product, while PyTorch still differentiates its coordinate expression.

```python
direction.grad = None
more_samples = torch.eye(3, dtype=torch.float64)
second_loss = inner(more_samples, direction).sum()
second_loss.backward()
torch.testing.assert_close(direction.grad, torch.ones_like(direction))

assert torch.autograd.gradcheck(
    inner,
    (samples.clone().requires_grad_(), direction.detach().requires_grad_()),
)
```

This second call has a different batch size and a fresh differentiation graph.
No replanning is required. `gradcheck` compares the local derivative with finite
differences; it is a check of this example, not a replacement for choosing an
appropriate loss or numerical scale.

## Reuse within PyTorch

Store a plan as an attribute of an `nn.Module` to register it as a child module.
Then ordinary `.to()` moves its execution buffers with the rest of that module.
Changing an algebra context's defaults affects future plans; existing plans
move independently. A new signature, layout, storage choice, or operation
requires a corresponding new plan.

Planning can reduce repeated setup, but it does not remove the cost of the
requested computation. Output width and the number of interacting blades
still matter, and broadcasting can create a large result. Request only the
grades the next computation needs.

The same tensor and differentiation conventions apply to the other planned
operations. [Bivector Exponentials and Induced Actions](exponentials-and-actions.md)
explores one geometric application. [Compute Products](../how-to/products.md)
covers further product operations.
