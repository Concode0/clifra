# Layers

## Primitives
::: clifra.layers.primitives.rotor.RotorLayer
::: clifra.layers.primitives.multi_rotor.MultiRotorLayer
::: clifra.layers.primitives.linear.CliffordLinear
::: clifra.layers.primitives.rotor_gadget.RotorGadget
::: clifra.layers.primitives.normalization.CliffordLayerNorm
::: clifra.layers.primitives.projection.BladeSelector

## Blocks
::: clifra.layers.blocks.attention.GeometricProductAttention
::: clifra.layers.blocks.multi_rotor_ffn.MultiRotorFFN
::: clifra.layers.blocks.transformer.GeometricTransformerBlock

## Adapters
::: clifra.layers.adapters.embedding.MultivectorEmbedding
::: clifra.layers.adapters.mother.MotherEmbedding
::: clifra.layers.adapters.mother.EntropyGatedAttention

!!! note "Optional dependency"
    `CliffordGraphConv` requires `torch-geometric`. Install with `uv sync --extra md17`.

::: clifra.layers.adapters.gnn.CliffordGraphConv
