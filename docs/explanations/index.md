# Explanations

These pages describe the mathematical representations and execution semantics
of the public library.

- [Layouts, Storage, and Tensor Contracts](layout-storage-contracts.md)
  defines basis order, compact coefficients, storage declarations, and axes.
- [Products, Grades, and Blade Geometry](products-and-geometry.md)
  states product and involution conventions and the domain of blade operations.
- [Signatures and Algebraic Behavior](signatures-and-learning.md)
  distinguishes signed forms from coefficient-space quantities and explains
  their gradients and conditioning.
- [Planned Execution and Resources](planning-policy-injection.md)
  separates fixed contracts, feasibility budgets, and executor selection.
- [Tensor Composition and PyTorch Ownership](clifra-and-pytorch.md)
  covers broadcasting, autograd, modules, placement, and compilation.

[Geometric Representations and Parameters](clifra-methodology.md) compares
vectors, blades, mixed-grade values, bivectors, and rotors.
[Optimization and Geometric Updates](optimizer-dispatch.md) starts from
ordinary coefficient updates and explains the assumptions of specialized
helpers.

[Bivector Exponentials and Actions](bivector-exponential.md) develops one
specialized operation in depth, including finite closures, numerical
evaluation, and induced actions. It builds on the same layouts and planning
contracts as the other operations.
