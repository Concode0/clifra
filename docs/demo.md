# Manifold Demo

The browser demo below mirrors the data-generation idea in
`examples/demo.py`: sample a distorted curve, measure the energy in the
out-of-plane lane, and reduce that energy to show the intended alignment
behavior.

<ManifoldDemo />

## Python Demo

The repository still includes the Streamlit version for local experiments with
the actual PyTorch layers:

```bash
uv sync --extra demo
uv run streamlit run examples/demo.py
```

The VitePress component is intentionally browser-native so the documentation
can be built and deployed as static files. The Python demo remains the place to
inspect the live `RotorLayer` and `BladeSelector` training loop.
