# Optimization for Geometric Parameters

Clifra separates optimization dynamics from geometric realization. A layout
defines the parameter coordinates, a layer turns those coordinates into a
geometric action, and an optimizer determines how the objective is explored.
This separation makes the built-in optimizers useful defaults without making
them the boundary of the optimization methods available to a Clifra model.

## Built-in parameter dispatch

`RiemannianAdam` and `ExponentialSGD` use ordinary PyTorch parameter groups with
one additional `manifold` field. Their `from_model()` constructors collect the
tags already attached by Clifra layers and apply the corresponding post-update
rule.

| Tag | Typical parameter | Post-update rule |
| --- | --- | --- |
| `spin` | Compact grade-2 generator coordinates | Apply the configured coefficient-norm guard. |
| `sphere` | Grade-1 reflection direction | Normalize by the signature magnitude, with a Euclidean fallback near null directions. |
| `euclidean` | Biases, mixing weights, and untagged parameters | Keep the optimizer's update unchanged. |

The `spin` parameter is already expressed in the Lie algebra. For

\[
B = \sum_{i<j} b_{ij} e_i e_j,
\]

a versor layer constructs

\[
R = \exp(-B/2)
\]

and applies the planned Clifford action during its forward pass. Adam or SGD
therefore optimizes compact, meaningful generator coordinates; the layer
realizes the group element. The optional norm guard keeps those coordinates in
a useful numerical range and can be disabled with `max_bivector_norm=None`.

```python
from clifra.optimizers import make_riemannian_optimizer

optimizer = make_riemannian_optimizer(
    model,
    algebra,
    optimizer="adam",
    lr=1e-3,
)
```

## Choose for the objective and parameter regime

Clifford parameterizations can make a transformation model compact enough that
methods designed for deterministic objectives or small parameter sets become
practical. The appropriate optimizer is a property of the problem, not of the
algebra alone.

| Regime | Practical starting point |
| --- | --- |
| Minibatches, noisy gradients, or many parameters | Adam-style coordinate optimization |
| Smooth training with simple memory requirements | SGD with momentum |
| Deterministic objectives with relatively few parameters | L-BFGS, nonlinear conjugate gradient, or another full-batch method |
| Objectives that benefit from curvature information | Newton, trust-region, or Hessian-vector methods |
| Parameters stored as rotors rather than generators | Tangent projection followed by an exponential update |

PyTorch's `LBFGS` is immediately usable when the training loop provides a
closure. Higher-order implementations can request a differentiable gradient
evaluation with `create_graph=True`, then use Hessian-vector products or a
problem-specific linear solve. These methods cost more per step, but that
tradeoff can be attractive for small Clifford fields and full-batch inverse
problems.

## Connect another optimizer

Clifra's tags are ordinary parameter-group metadata, so an optimizer adapter
can follow the same structure as the built-ins:

1. Use `group_parameters_by_manifold(model)` to collect coordinate groups.
2. Pass those tensors to the selected optimizer, preserving the `manifold`
   value on each parameter group when the adapter uses it.
3. Perform the optimizer's coordinate update.
4. Apply the tag-specific post-update rule required by that parameterization.

For a generator-only model, a standard PyTorch or third-party optimizer may be
used directly: the layer's exponential still constructs the rotor, and the
`spin` coefficient guard is optional. A model containing `sphere` parameters
needs an adapter that restores their normalization after each update. This is
the same division of responsibility implemented by `RiemannianAdam` and
`ExponentialSGD`.

If a method stores full-lane unit rotors directly, the public
`project_to_tangent_space` and `exponential_retraction` functions demonstrate
the complementary manifold update:

\[
P_R(V) = R\left\langle \widetilde{R}V \right\rangle_2,
\qquad
\operatorname{Exp}_R(RB) = R\exp(B).
\]

These functions use canonical full-lane rotors at their boundary and compact
grade-2 lanes internally. They are building blocks for an optimizer adapter;
Clifra's bivector-parameterized layers do not need to store or renormalize a
rotor parameter themselves.

Optimizer tags should remain about parameter-level update rules. Invariance,
field constraints, and application-specific structure belong in layouts,
layers, actions, or objectives, where they can be stated and tested directly.
