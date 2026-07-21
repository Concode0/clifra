# Explanations

These chapters develop the design of clifra from tensor representation through
geometric parameterization, numerical execution, and PyTorch integration.

1. [Layout, Storage, and `TensorContract`](layout-storage-contracts.md) explains
   how clifra represents a selected set of blades as an ordinary dense tensor.
2. [Planning Policy as Dependency Injection](planning-policy-injection.md)
   explains resource limits and executor selection.
3. [Geometric Parameterization](clifra-methodology.md) describes geometric
   objects as learnable coordinate systems and positions the physics-informed
   bivector-field showcase within that method.
4. [Why Bivector Coordinate Fields Work](transformation-fields.md) derives the
   generalized input, sampling, action, and inversion contracts behind the
   continuum-solver research package.
5. [Bivector Exponential Methods](bivector-exponential.md) separates
   exact low-dimensional formulas, matrix exponentiation, and spectral-local
   approximation.
6. [Optimization for Geometric Parameters](optimizer-dispatch.md) explains the
   built-in parameter dispatch and how to connect quasi-Newton, higher-order,
   or tangent-space methods.
7. [Signatures and Algebraic Behavior](signatures-and-learning.md) distinguishes
   signed Clifford forms from positive coefficient-lane energy and relates both
   to differentiation.
8. [Using clifra with PyTorch](clifra-and-pytorch.md) explains how clifra's
   algebraic machinery and the surrounding PyTorch system divide the work.

The chapters are independent of the tutorials. Each introduces the definitions
needed for its own argument; the [API reference](../reference/index.md) provides
the corresponding public interfaces.
