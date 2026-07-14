# Explanations

These chapters describe the design decisions behind clifra, the mathematical
boundaries of its execution routes, and the distinction between algebraic
structure and application policy.

1. [Layout, Storage, and `TensorContract`](layout-storage-contracts.md) explains
   how clifra represents a selected set of blades as an ordinary dense tensor.
2. [Planning Policy as Dependency Injection](planning-policy-injection.md)
   explains resource limits and executor selection.
3. [Geometric Parameterization](clifra-methodology.md) describes geometric objects as
   learnable coordinate systems and places the continuum solver in that method.
4. [Bivector Exponential Methods](bivector-exponential.md) separates
   exact low-dimensional formulas, matrix exponentiation, and spectral-local
   approximation.
5. [Optimizer Parameter Categories](optimizer-dispatch.md) explains the three
   parameter tags and the role of bivector coordinates.
6. [Signatures and Algebraic Behavior](signatures-and-learning.md) distinguishes
   signed Clifford forms from positive coefficient-lane energy and relates both
   to differentiation.
7. [Using clifra with PyTorch](clifra-and-pytorch.md) defines the boundary between
   clifra's algebraic machinery and the surrounding PyTorch system.

The chapters are independent of the tutorials. Each introduces the definitions
needed for its own argument; the [API reference](../reference/index.md) provides
the corresponding public interfaces.
