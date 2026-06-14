# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Minimal scalar metric logging for continuum solver runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .types import MetricValue


@dataclass(frozen=True)
class MetricRecord:
    """One logged optimization step."""

    step: int
    metrics: Mapping[str, MetricValue]


@dataclass
class MetricLogger:
    """Append-only scalar metric store."""

    records: list[MetricRecord] = field(default_factory=list)

    def append(self, step: int, metrics: Mapping[str, MetricValue]) -> None:
        """Append one step of scalar metrics."""
        self.records.append(MetricRecord(step=int(step), metrics=dict(metrics)))

    def latest(self) -> MetricRecord | None:
        """Return the most recent record, if any."""
        return self.records[-1] if self.records else None

    def keys(self) -> tuple[str, ...]:
        """Return all metric keys seen in insertion order."""
        ordered: dict[str, None] = {}
        for record in self.records:
            for key in record.metrics:
                ordered.setdefault(key, None)
        return tuple(ordered)

    def series(self, key: str) -> list[tuple[int, MetricValue]]:
        """Return ``(step, value)`` pairs for a metric key."""
        return [(record.step, record.metrics[key]) for record in self.records if key in record.metrics]

    def rows(self) -> list[dict[str, MetricValue | int]]:
        """Return records as flat dictionaries for scripts that want tabular output."""
        return [{"step": record.step, **record.metrics} for record in self.records]
