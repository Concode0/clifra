# Tutorials

The first three tutorials introduce Clifford coefficient tensors, layouts,
ordinary tensor composition, and reusable operations with autograd. Each page
is self-contained; run its Python blocks in order. Python and PyTorch are
assumed.

1. [First Clifford Product](first-clifford-product.md) introduces the signature,
   coefficient lanes, and the scalar and exterior parts of a vector product.
2. [Layouts, Storage, and Tensor Composition](layouts-and-plans.md) separates
   coefficient meaning from physical storage and leading tensor dimensions.
3. [Planned Differentiation](planned-differentiation.md) fixes an operation's
   contracts and reuses it with changing values and gradients.
Three optional application studies use those same contracts and tensor
operations for geometric transformations:

4. [Bivector Exponentials and Induced Actions](exponentials-and-actions.md)
   compares a materialized rotor with its action on vectors.
5. [Fit a Rotation with PyTorch](learn-geometric-transform.md) fits a
   rotation from point correspondences with an ordinary PyTorch parameter.
6. [Differentiate a Spatial Deformation](unbend-manifold.md) composes a
   coordinate-dependent action with PyTorch to fit a twisted surface from
   sparse observations.

[Explanations](../explanations/index.md) develop the mathematical and numerical
semantics. The [API Reference](../reference/index.md) describes the public API.
