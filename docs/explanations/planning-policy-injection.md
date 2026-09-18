# Planned Execution and Resources

A product depends on more than the two coefficient tensors. Its signature,
input layouts, product convention, and output projection determine which basis
interactions contribute. Planning resolves that structure before evaluating
coefficients.

For fixed layouts, a product has the form

\[
y_k=\sum_{i,j}c_{ijk}a_i b_j.
\]

The coefficients \(c_{ijk}\) include basis permutation signs, metric factors,
and the requested grade selection. Terms known to vanish need not be evaluated.
The leading tensor dimensions are independent of this coefficient structure.

## What a plan fixes

Every public `plan_*` method returns a `PlannedOperation`. Its `inputs` and
`output` properties hold the declared tensor contracts. Planning fixes:

- the signature and mathematical operation;
- input and output layouts and storage forms;
- ordinary broadcasting or explicit pairwise item axes, for products;
- the selected implementation and its initial dtype and device.

Tensor values and leading dimension sizes remain runtime inputs. A product
planned for compact vectors can process one vector, a batch, or broadcast
collections without changing its algebraic contract.

```python
import torch
from clifra import make_algebra

algebra = make_algebra(3, 0)
vectors = algebra.layout((1,))
product = algebra.plan_product(left=vectors, right=vectors)

assert product.output.layout.grades == (0, 2)
assert product(torch.ones(3), torch.ones(3)).shape == (4,)
assert product(torch.ones(5, 1, 3), torch.ones(7, 3)).shape == (5, 7, 4)
```

Calling a retained operation converts declared storage, validates operand
boundaries, and executes its tensor program directly, independent of the
algebra and planner. A plan is an `nn.Module`, so it can own buffers, move with
`.to()`, and be registered as a child of another module.

The execution method is fixed, but its numerical program can contain
value-dependent work. Taylor exponentiation, for example, chooses a scaling
count from the generator magnitude. This is different from selecting a new
operation or changing its declared output.

## Direct calls, caches, and ownership

Direct calls such as `algebra.geometric_product(...)` use the same planning
machinery and per-algebra caches. Use them for an eager expression; retain a
plan when a repeated computation should have an explicit module boundary.

`algebra.to(...)` changes defaults for future plans and clears its planning
caches. Existing plans move independently, each retaining its selected
implementation. Moving one plan leaves the placement of other plans unchanged,
even when they originally shared cached work. Construct a new plan when
selection should be reconsidered for another dtype or device.

Private executor types, route names, and cost formulas are not plan
inspection APIs. The stable result of planning is the callable operation and
its contracts. Save the declarations needed to reconstruct it; private kernel
state is not a portable serialization format across library versions.

## What resource checks measure

Compact storage removes coefficients absent from the layout, but an operation
can still cost more than the output width. A full layout has \(2^n\)
lanes; a grade-\(k\) layout has \(\binom nk\). An operation can also require many
input interactions or a larger intermediate layout.

Planning checks both maximum lane width and a conservative static
pair/interaction footprint before building execution buffers. The footprint
includes route-owned structures and child computations that coexist, and may
include a larger known fixed-shape temporary. It is not a runtime byte count.

`ResourceLimits` is an immutable feasibility budget supplied by the caller.
The defaults warn at 2,048 lanes or 1,000,000 interactions and reject
requirements above 4,096 lanes or 8,000,000 interactions. Supply another budget
at construction:

```python
from clifra import ResourceLimits

limits = ResourceLimits(max_lanes=512, max_pairs=100_000)
bounded = make_algebra(3, resource_limits=limits)
assert bounded.resource_limits is limits
```

The same argument is available on `AlgebraContext` and `AlgebraConfig`.
The algebra owns the supplied budget for its lifetime, and placement changes
preserve it. Construct another algebra to use a different budget.
See [Set resource limits](../how-to/resource-limits.md) for examples.

Providers first assess conservative resource requirements. The router compares
them with the budget; only feasible built-in candidates reach performance
ranking. Raising a limit permits a computation, but performance ranking remains separate.
Budgets define resource feasibility; they do not alter the operation's mathematics or request an approximation.

Caller-controlled batch and item dimensions are excluded from these static
counts. A legal pairwise product can still create an output proportional to
\(L R\), and its backward pass may retain additional tensors. Size or chunk
those dimensions in the application.

Limits reject unavailable computations; they do not request a lower-rank approximation.
Narrowing the output changes the requested projection, whereas
changing a registry changes the available implementations.

Private configuration and harness utilities serve kernel tests and benchmark
infrastructure. They are not user-facing tuning interfaces. Public budget
injection does not expose their policy or configuration machinery.

## Executor extensions

An optional `ExecutorRegistry` supplies implementations through the public
`ExecutorProvider` protocol. The provider receives an `ExecutorRequest`:
operation family and name, compact contracts, dtype, and device. A `None`
input contract denotes an ordinary tensor operand, such as a matrix.

Assessment and construction are separate. `assess` returns either a rejection
reason or an `Assessment` containing conservative resource requirements and
optional provider-owned preparation. Assessment must not allocate execution
buffers. If selected, `build` receives the same request and assessment objects
and returns an `nn.Module`.

Acceptance means implementing the declared mathematical operation, allowing
ordinary numerical roundoff but no deliberate truncation. The module must
honor broadcasting, device and dtype behavior, and differentiation through its
tensor computation. clifra supplies compact/canonical conversions and pairwise
expansion at the boundary.

A supplied registry replaces the default registry. Prepending an external
provider to the default providers gives it precedence when eligible; appending
one makes it a fallback. Precedence applies only to accepted providers; rejected and over-budget
providers are skipped. Provider identities and behavior remain fixed while a registry is
in use. There is no global registration state.

[Provide a custom executor](../how-to/custom-executors.md) gives a complete
provider example without importing private planning structures.
