"""Repository-private performance measurement tools for clifra."""

from __future__ import annotations

__all__ = [
    "BenchmarkCase",
    "BenchmarkAdapter",
    "BenchmarkRequest",
    "ConstructedBenchmark",
    "ExecutionPlacement",
    "LayoutCase",
    "MeasurementConfig",
    "OrdinaryTensorCase",
    "PreparedBenchmark",
    "SelectionCase",
    "measure_request",
    "smoke_cases",
]


def __getattr__(name: str):
    if name in {
        "BenchmarkCase",
        "ExecutionPlacement",
        "LayoutCase",
        "OrdinaryTensorCase",
        "SelectionCase",
        "smoke_cases",
    }:
        from . import cases

        return getattr(cases, name)
    if name == "MeasurementConfig":
        from . import engine

        return getattr(engine, name)
    if name == "measure_request":
        from . import engine

        return engine.measure_request
    if name in {"BenchmarkAdapter", "BenchmarkRequest", "ConstructedBenchmark", "PreparedBenchmark"}:
        from . import model

        return getattr(model, name)
    raise AttributeError(name)
