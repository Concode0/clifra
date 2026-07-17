# API Reference

This page is generated from clifra's source docstrings. Public documented
modules, classes, methods, functions, and attributes are included once from
their defining modules. Private names and undocumented implementation details
are omitted.

## Algebra and Runtime

::: clifra.core.config

::: clifra.core.formatting

::: clifra.core.foundation.basis

::: clifra.core.foundation.device

::: clifra.core.foundation.host

::: clifra.core.foundation.layout

::: clifra.core.foundation.manifold

::: clifra.core.foundation.module

::: clifra.core.foundation.numerics

::: clifra.core.foundation.validation

::: clifra.core.runtime.algebra

::: clifra.core.runtime.energy

::: clifra.core.runtime.forms

::: clifra.core.runtime.metric

::: clifra.core.runtime.tensors

## Planning

::: clifra.core.planning.action

::: clifra.core.planning.exp

::: clifra.core.planning.flow

::: clifra.core.planning.layouts

::: clifra.core.planning.metric

::: clifra.core.planning.permutation

::: clifra.core.planning.planner

::: clifra.core.planning.policy

::: clifra.core.planning.product

::: clifra.core.planning.tree

::: clifra.core.planning.unary

## Execution

::: clifra.core.execution.action

::: clifra.core.execution.attention

::: clifra.core.execution.exp

::: clifra.core.execution.handles

::: clifra.core.execution.metric

::: clifra.core.execution.permutation

::: clifra.core.execution.product

::: clifra.core.execution.unary

## Functional Operations and Criteria

::: clifra.functional.activation

::: clifra.functional.loss

::: clifra.functional.orthogonality

::: clifra.functional.products

::: clifra.criterion.loss

::: clifra.criterion.orthogonality

## Layers

::: clifra.layers.adapters.conformal

::: clifra.layers.adapters.projective

::: clifra.layers.blocks.attention

::: clifra.layers.primitives.activation

::: clifra.layers.primitives.linear

::: clifra.layers.primitives.multi_versor

::: clifra.layers.primitives.normalization

::: clifra.layers.primitives.product

::: clifra.layers.primitives.projection

::: clifra.layers.primitives.reflection

::: clifra.layers.primitives.rotor_gadget

::: clifra.layers.primitives.versor

## Optimization

::: clifra.optimizers.riemannian

## Analysis

These interfaces provide experimental geometric diagnostics. Their outputs
describe the implemented tensor and coefficient-space calculations; they are
not statistical inference or proofs of causal, metric, or symmetry structure.

::: clifra.analysis
    options:
      members:
        - AnalysisConstants
        - SamplingConfig
        - AnalysisConfig
        - DimensionResult
        - SignatureEstimate
        - SpectralResult
        - TransformationDiagnosticsResult
        - CommutatorResult
        - AnalysisReport

::: clifra.analysis.commutator

::: clifra.analysis.dimension

::: clifra.analysis.geodesic

::: clifra.analysis.pipeline

::: clifra.analysis.policy

::: clifra.analysis.sampler

::: clifra.analysis.signature

::: clifra.analysis.spectral

::: clifra.analysis.symmetry

## Utilities

::: clifra.utils.mps
