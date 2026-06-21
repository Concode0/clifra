# clifra

Clifra is documented from the package outward: API pages render module
docstrings, while the core design page only keeps the small process snippets
needed to connect those APIs.

Start with [Core Design](core-design.md), then use the [API](api/index.md).

See [first-guide](first-guide.md), If you are a little confused or think you need an intuitive understanding, please refer to the first guide and move on to the [API](api/index.md).

---

## Performance Foundation

The maximum-density verification matrix and all performance profiles rendered in benchmarks section were captured on a baseline workstation to ensure reproducible benchmarks:

- **Hardware:** Apple MacBook Pro (Apple M5 Pro, 48GB Unified Memory)
- **Execution Target:** Purposely restricted to Host CPU (`torch_num_threads=5`) to isolate core algorithmic efficiency from raw GPU tensor core acceleration.
- **Framework:** PyTorch `2.10.0` on `macOS` (ARM64)

For comprehensive dimension sweeps up to $Cl(63)$ and accumulated drift topologies, see the [Detailed Benchmarks](benchmarks/index.md)