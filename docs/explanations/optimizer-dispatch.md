# Optimizer Parameter Categories

Clifra's tag-aware optimizers recognize only three parameter tags:
`spin`, `sphere`, and `euclidean`. These tags are update and retraction metadata.
They describe optimizer behavior. Classification of every geometric object that
a Clifford algebra can represent is a separate concern.

The limited dispatch is deliberate. Most geometric structure belongs in a
layer's parameterization and forward action. The optimizer only handles the
constraint that remains after its ordinary Adam or SGD update.

## The three update categories

| Tag | Typical parameter | Post-update behavior |
| --- | --- | --- |
| `spin` | Grade-2 bivector coordinates | Clip the Euclidean coefficient norm to `max_bivector_norm`, if configured. |
| `sphere` | Grade-1 reflection direction | Normalize by the signed signature magnitude when non-null; use a Euclidean fallback near null values. |
| `euclidean` | Biases, mixing weights, and untagged parameters | No geometric retraction. |

`group_parameters_by_manifold` puts every untagged parameter in the Euclidean
group. `RiemannianAdam.from_model` and `ExponentialSGD.from_model` preserve the
usual optimizer parameter-group model while dispatching the post-update step by
tag.

The layer constructs the rotor; the optimizer's `spin` branch only applies a
numerical norm cap after Adam or SGD updates the bivector coefficients. During
the forward pass, the layer maps those coefficients into a rotor through the
planned bivector exponential.

The `sphere` branch checks whether the parameter width matches the algebra's
grade-1 layout. If it does, normalization uses the magnitude of the signed form.
A near-null vector cannot be safely normalized by that form, so the branch uses
its positive Euclidean coefficient norm. If the width is not a grade-1 layout,
the same Euclidean fallback is used.

## Bivector parameterization

The spin group is parameterized in the Lie algebra: a layer stores a bivector

\[
B = \sum_{i<j} b_{ij} e_i e_j
\]

and constructs

\[
R = \exp(-B/2)
\]

during the forward pass. The action on an input is then built from $R$, its
reverse, and Clifford products.

The parameterization has the following properties:

- the parameter tensor has a fixed grade-2 layout;
- updates and optimizer moments live in an unconstrained linear coordinate
  space;
- the forward map constructs a valid versor action rather than requiring an
  arbitrary coefficient tensor to remain rotor-normalized;
- learned coefficients describe plane generators, so the parameter itself has
  geometric meaning.

The entire geometric object is learned through its coordinates. Because the
exponential and action already encode rotor dimension and plane composition,
the optimizer can use the same case for all of them.

## When to add optimizer tags

An optimizer tag should answer a narrow question: what correction is required
immediately after updating this parameter? Many useful objects already fit the
existing categories.

A mixed-grade feature can be an ordinary Euclidean parameter even though its
forward interpretation is geometric. A constrained field may need a loss or a
domain-specific projection rather than a universal optimizer retraction. An
invariant readout belongs to the layer composition and calls for no new kind of
parameter update.

Adding tags for these cases would move model semantics into optimizer dispatch
and overstate the guarantees available from a local update. A new category is
justified when a distinct parameter-level constraint has a well-defined,
reusable retraction.

## Parameterization limits

Exponential coordinates can be many-to-one: different bivectors may produce the
same or equivalent group action, and large coordinates can create poorly
conditioned optimization paths. The spin norm cap is a numerical guard rather
than an exact Riemannian exponential update, and global coordinate ambiguity
remains.

Likewise, a versor layer alone guarantees neither invariance nor isometry for an
entire model. Euclidean channel mixing, nonlinearities, readouts, losses, and
numerical approximations can change the complete model's behavior. Required
invariants must be tested at the level where they are claimed.

The optimizer remains simple because it is the final part of a larger design:
the layout chooses the coordinate space, the layer constructs the geometric
action, and the optimizer applies only the minimal parameter-level correction.
