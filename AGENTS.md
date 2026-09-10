# Repository Guidelines

## Project Structure & Module Organization

`clifra/core/` is the source of truth for Clifford algebra semantics, layouts, tensor contracts, planning, resource feasibility, and execution.

Keep the core model explicit:

```text
mathematical request
→ planning
→ feasible route
→ selected executor
→ tensor execution
```

Planning owns route selection; executors execute fixed prepared tensor computations and must not re-plan.

`clifra/analysis/` and `clifra/optimizers/` build on core contracts rather than duplicating algebra semantics. Exploratory analysis belongs in `clifra/analysis/experimental/`.

Documentation lives in `docs/`; runnable demonstrations and research systems live in `research/`; internal performance and qualification tooling lives under `benchmarks/`.

## Build, Test, and Development Commands

Use `uv` for project tooling.

```bash
uv sync --group dev

uv run --group dev pytest tests/ -n12 -q --tb=short
uv run --group dev pytest tests/planning/ -n12 -q --tb=short
uv run --group dev pytest tests/ --hypothesis-profile=full -n12 -q --tb=short

uv run --group dev ruff check .
uv run --group dev ruff format .
uv run --group docs mkdocs build
```

Do focused tests while iterating, then run the full suite before handing off cross-cutting changes.

## Coding Style & Architecture

Ruff is the source of truth for Python linting conventions.

Preserve explicit layouts, compact/canonical storage, and tensor contracts. The final axis contains Clifford coefficients; leading axes remain ordinary PyTorch dimensions and broadcasting.

Do not infer algebraic meaning from tensor width, channels, names, or application structure.

Keep `DefaultPolicy` small, deterministic, structural, and device-independent. Benchmarks may reveal broad route trends, but must not introduce runtime calibration, fitted cost models, backend-specific tuning tables, or autotuning.

Prefer small local executor optimizations over new abstraction layers. Preserve mathematical behavior, autograd, dtype/device semantics, broadcasting, and compile compatibility.

Tests should protect mathematical and public behavior, not private cache identities, exact planner metadata, or historical implementation structure.

When benchmark or backend qualification exposes a real correctness issue, reduce it to a focused test.
