# Architecture

Versor is organized around one rule: algebraic identity, tensor layout, runtime
execution, and training behavior should stay separable. The core owns the
mathematics and the execution contracts; layers, functionals, and optimizers
expose those contracts in forms that are convenient for actual model code.

## Framework Layers

```mermaid
flowchart TB
    User["User code<br/>layers, functionals, optimizers"]

    subgraph Framework["Versor framework"]
        Layers["layers/<br/>nn.Module composition"]
        Functional["functional/<br/>stateless algebra helpers, losses, activations"]
        Optimizers["optimizers/<br/>manifold-aware parameter updates"]
    end

    subgraph Core["core/"]
        Foundation["foundation/<br/>basis rules, layouts, validation, manifold tags"]
        Planning["planning/<br/>grade requests, lane limits, executors"]
        Runtime["runtime/<br/>dense algebra, planned context, multivectors"]
        Config["config.py<br/>algebra factory"]
        Analysis["analysis/<br/>metric/geodesic/symmetry tooling"]
    end

    User --> Layers
    User --> Functional
    User --> Optimizers
    Layers --> Runtime
    Layers --> Functional
    Layers --> Foundation
    Functional --> Runtime
    Optimizers --> Foundation
    Optimizers --> Runtime
    Config --> Runtime
    Runtime --> Planning
    Runtime --> Foundation
    Planning --> Foundation
    Analysis --> Runtime
```

`core/` is the authority for algebra and execution. `layers/` should be thin
module wrappers around core operations. `functional/` should expose stateless
helpers and losses without duplicating runtime logic. `optimizers/` should read
parameter manifold tags and apply updates; it should not decide product plans.

## Algebra Host Selection

```mermaid
flowchart LR
    Factory["make_algebra(p, q, r, kernel)"]
    Dense["CliffordAlgebra<br/>dense tables, full dim = 2^n"]
    Context["AlgebraContext<br/>planned compact execution"]
    Spec["AlgebraSpec<br/>signature and dimension metadata"]
    Planner["GradePlanner<br/>cached layouts and executors"]

    Factory -->|"small n or explicit dense"| Dense
    Factory -->|"large n or explicit context"| Context
    Dense --> Spec
    Context --> Spec
    Dense --> Planner
    Context --> Planner
    Planner --> Spec
```

`CliffordAlgebra` owns dense Cayley-table buffers and fast full-layout kernels.
`AlgebraContext` exposes the same high-level product API but routes products
through static grade planning by default. Both hosts share the runtime facade, so
declared products use the same layout, cost, and executor path.

## Tensor And Layout Flow

```mermaid
flowchart LR
    Grades["Grade set<br/>for example (1, 2)"]
    Basis["basis_index_tuple_for_grades(n, grades)"]
    Layout["GradeLayout<br/>grades, dense basis indices, lane count"]
    Dense["Dense multivector<br/>last dim = 2^n"]
    Compact["Compact values<br/>last dim = layout.dim"]
    Materialized["Dense materialization<br/>zeros outside layout"]

    Grades --> Basis --> Layout
    Dense -->|"layout.compact"| Compact
    Compact -->|"layout.dense"| Materialized
    Layout --> Compact
    Layout --> Materialized
```

A compact tensor is not just a shorter tensor. It is coefficient values plus a
`GradeLayout` identity that defines which dense basis blades the lanes
represent. Raw tensors do not carry that identity, so framework pipeline code
must declare `*_grades`, pass layouts, or use `Multivector` wrappers when layout
metadata needs to travel with values.

## Product Execution Flow

```mermaid
flowchart TB
    A["A tensor<br/>dense [*, 2^n] or compact [*, left_lanes]"]
    B["B tensor<br/>dense [*, 2^n] or compact [*, right_lanes]"]
    Metadata["Declared metadata<br/>left_grades, right_grades, output_grades<br/>left_layout, right_layout, compact flags"]

    Request["ProductRequest<br/>normalized op, layouts, dtype, device"]
    Cost["PlanningLimits and cost checks<br/>lanes, output width, estimated pairs"]
    Tree["GradePlanTree<br/>homogeneous route nodes"]
    Plan["GradeProductPlan<br/>left lanes, right lanes, output lanes, coefficients"]
    Exec["GradeProductExecutor<br/>forward, forward_compact, forward_pairwise_compact"]
    Values["Compact output values<br/>[*, output_lanes]"]
    DenseOut["Dense materialized output<br/>[*, 2^n]"]

    A --> Request
    B --> Request
    Metadata --> Request
    Request --> Cost
    Cost --> Tree
    Tree --> Plan
    Plan --> Exec
    A --> Exec
    B --> Exec
    Exec --> Values
    Values -->|"compact_output=True"| CompactReturn["Return compact values"]
    Values -->|"compact_output=False"| DenseOut
```

Planning uses static grade metadata and tensor shapes. It does not inspect
runtime tensor values. This keeps compiled paths stable and avoids
data-dependent symbolic shape extraction.

## Layer Pipeline Contract

```mermaid
sequenceDiagram
    participant Model as Model/Layers
    participant ProductLayer as ProductLayer
    participant Algebra as Algebra Runtime API
    participant Planner as GradePlanner
    participant Executor as GradeProductExecutor
    participant Optimizer as Optimizer

    Model->>ProductLayer: forward(left, right)
    ProductLayer->>Algebra: projected_product(...grades, compact flags)
    Algebra->>Planner: product_request(...)
    Planner->>Executor: cached or newly built executor
    Algebra->>Executor: forward / forward_compact / forward_pairwise_compact
    Executor-->>Model: dense or compact tensor
    Model->>Optimizer: loss.backward(); step()
    Optimizer->>Optimizer: update tagged parameters
```

Optimizers do not run planning. They update parameters after the forward pass.
Planning happens when the model calls algebra operations through direct runtime
APIs, functional helpers, `ProductLayer`, or `Multivector` methods.

## Operator Rules

```mermaid
flowchart TB
    Route["Route<br/>left grade r, right grade s"]
    GP["geometric product<br/>grades abs(r-s), abs(r-s)+2, ..."]
    Wedge["wedge / exterior<br/>grade r+s only"]
    Symmetric["inner route<br/>symmetric parity grades"]
    Comm["commutator<br/>odd swap-parity grades"]
    Anti["anti-commutator<br/>even swap-parity grades, doubled coefficient"]
    Pair["Basis pair<br/>left index i, right index j"]
    Output["output index = i XOR j"]
    Nonzero["operation_may_be_nonzero<br/>wedge requires no shared basis bit<br/>null self-overlap is zero"]
    Coeff["operation_coefficient<br/>metric sign and op scale"]

    Route --> GP
    Route --> Wedge
    Route --> Symmetric
    Route --> Comm
    Route --> Anti
    GP --> Pair
    Wedge --> Pair
    Symmetric --> Pair
    Comm --> Pair
    Anti --> Pair
    Pair --> Output
    Output --> Nonzero
    Nonzero --> Coeff
```

The wedge implementation is the exterior product. For homogeneous inputs:

```text
A_r ^ B_s = <A_r B_s>_{r+s}
```

For vectors this coincides with `(AB - BA) / 2`, but higher-grade wedge routes
follow the grade-sum exterior definition.

## Dense Runtime

```mermaid
flowchart TB
    Init["CliffordAlgebra init"]
    Cayley["Cayley indices and signs<br/>basis_product over all pairs"]
    GP["gp_signs<br/>single-pass geometric product"]
    Wedge["wedge_gp_signs<br/>grade(output) = grade(left) + grade(right)"]
    Inner["inner_gp_signs<br/>symmetric sign table"]
    Comm["comm_gp_signs and anti_comm_gp_signs"]
    Contractions["left and right contraction helpers"]
    Product["Runtime product call"]
    Matmul["gather B by Cayley indices<br/>multiply signs<br/>batched matmul over lanes"]

    Init --> Cayley
    Cayley --> GP
    Cayley --> Wedge
    Cayley --> Inner
    Cayley --> Comm
    Cayley --> Contractions
    GP --> Product
    Wedge --> Product
    Inner --> Product
    Comm --> Product
    Product --> Matmul
```

Dense products are table-driven. The input tensors carry coefficients; the
precomputed tables carry basis multiplication structure.

## Planning Limits

`PlanningLimits` centralizes static guardrails for compact planning:

```python
from clifra.core.planning import PlanningLimits
from clifra.core.runtime.context import AlgebraContext

limits = PlanningLimits(max_lanes=8192, max_pairs=16_000_000)
algebra = AlgebraContext(32, 0, device="cpu", planning_limits=limits)
```

`max_lanes` protects compact tensor width. `max_pairs` protects the
gather/reduce interaction count generated by product plans. Dense algebra hosts
and planned contexts both accept `planning_limits`, so the same policy object can
be used across framework construction.

## Analysis Flow

```mermaid
flowchart LR
    Data["Input data<br/>vectors or multivectors"]
    Algebra["Algebra host<br/>dense or context"]
    Metric["Metric and dimension analysis<br/>norms, distances, effective dimension"]
    Geodesic["GeodesicFlow<br/>grade-1 wedge to connection bivectors"]
    Symmetry["SymmetryDetector<br/>group and continuous symmetry"]
    Comm["CommutatorAnalyzer<br/>bracket spectra and exchange structure"]
    Report["Analysis result dataclasses"]

    Data --> Algebra
    Algebra --> Metric
    Algebra --> Geodesic
    Algebra --> Symmetry
    Algebra --> Comm
    Metric --> Report
    Geodesic --> Report
    Symmetry --> Report
    Comm --> Report
```

Analysis code should call algebra APIs rather than reconstructing basis rules.
When active grades are known, analysis should pass `left_grades`,
`right_grades`, and `output_grades` so the planner can avoid full-layout work.

## End-To-End Product Call

```mermaid
sequenceDiagram
    participant Caller as Caller
    participant Algebra as Algebra Runtime API
    participant Planner as GradePlanner
    participant Executor as GradeProductExecutor
    participant Layout as GradeLayout

    Caller->>Algebra: wedge(A, B, left_grades, right_grades, output_grades)
    Algebra->>Planner: product_request(...)
    Planner->>Layout: resolve operand and output layouts
    Planner->>Planner: build or fetch cached executor
    Planner->>Executor: return executor
    Algebra->>Executor: forward, forward_compact, or forward_pairwise_compact
    Executor-->>Algebra: compact output values
    Algebra->>Layout: materialize dense unless compact_output=True
    Algebra-->>Caller: tensor output
```

The same path applies to geometric product, wedge, inner route, commutator, and
anti-commutator. The operator name changes route grades and coefficients, not
the overall tensor plumbing.

## Framework Verification Map

Framework-level tests are grouped by the behavior they prove:

- Core algebra and dense kernel identities: `tests/test_core.py`
- Static grade planning and compact execution: `tests/test_grade_plan.py`
- Multivector layout-preserving wrappers: `tests/test_multivector.py`
- Layer pipeline and optimizer integration: `tests/test_framework_pipeline.py`
- Functional product helpers: `tests/test_functional_products.py`
- Optimizer manifold grouping and factories: `tests/test_riemannian_optimizer.py`

Performance checks live in `benchmarks/`. The framework pipeline benchmark
measures dense-vs-compact products, planned contexts, pairwise compact products,
and composed layer pipelines.
