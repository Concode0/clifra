# clifra

Clifra provides layout-first Clifford algebra tools for PyTorch. It represents
geometric values as tensors, plans algebra operations from explicit grade
layouts, and exposes the resulting work as reusable PyTorch modules. It can be
used as a Clifford algebra library, a geometric deep-learning toolkit, or the
foundation of a separate research or domain-specific system.

## Choose a Path

| I want to... | Start here |
| --- | --- |
| learn clifra from the beginning | [First Clifford Product](tutorials/first-clifford-product.md) |
| understand layouts, planning, or learnable geometry | [Explanations](explanations/index.md) |
| look up a public interface or tensor contract | [API Reference](reference/index.md) |
| inspect performance and numerical profiles | [Benchmarks](benchmarks/index.md) |

## Benchmark Suite

The configured benchmark separates full-layout measurements through dimension
8 from compact-layout measurements through dimension 63. It records setup,
cold-call, forward and backward timing distributions, throughput, tensor
statistics, cumulative error, and the complete execution context.

See [Benchmarks](benchmarks/index.md) for the matrix, commands, graphs, and raw
artifacts.

## Research Demonstration

The continuum solver brings Clifra's full methodology together: semantic grade
layouts, planned Clifford actions, differentiable execution, sampled bivector
generators, and objective-driven learning. Its compact example demonstrates the
general transformation field, while the physics-informed example develops that
field into a validated deformation system.

See [Bivector Coordinate Fields](explanations/transformation-fields.md) for the
derivation and the exact sampling, path, and inversion semantics.
