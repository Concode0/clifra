# Examples

The examples below are checked during documentation verification. They are
small CPU scripts and should run from the repository root.

## Quickstart Model

```bash
uv run python docs/examples/quickstart.py
```

This verifies the basic dense path: vector embedding, rotor action, channel
mixing, activation, and grade selection.

## Products and Layouts

```bash
uv run python docs/examples/products_and_layouts.py
```

This verifies both dense products and compact layout planning.

## Training Step

```bash
uv run python docs/examples/training_step.py
```

This verifies that a small model can run forward, compute a geometric loss,
backpropagate, and step the Riemannian optimizer.
