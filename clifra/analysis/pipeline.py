# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Orchestrate descriptive and experimental diagnostics in one pipeline."""

import concurrent.futures
import time
from typing import Optional

import torch

from clifra.core.config import make_algebra
from clifra.core.foundation.module import AlgebraLike

from ._types import CONSTANTS, AnalysisConfig, AnalysisReport
from ._utils import as_analysis_tensor
from .commutator import CommutatorAnalyzer
from .dimension import CovarianceDimensionAnalyzer
from .sampler import StatisticalSampler
from .signature import SignatureProbeAnalyzer
from .spectral import SpectralAnalyzer
from .symmetry import TransformationDiagnosticsAnalyzer


class GeometricAnalyzer:
    """Top-level orchestrator for the experimental analysis toolkit.

    Runs a configurable subset of analyzers in the correct dependency
    order and returns an :class:`AnalysisReport`.

    **Input modes:**

    * ``data.ndim == 2`` and ``algebra is None`` -- **raw mode**: full
      pipeline from sampling through signature estimation to GA diagnostics.
    * ``data.ndim == 3`` and ``algebra is not None`` -- **pre-embedded
      mode**: skip sampling, dimension, and signature estimation; run spectral,
      transformation, and commutator diagnostics directly.
    * ``data.ndim == 2`` and ``algebra is not None`` -- **raw + known
      algebra**: embed data, then run GA analyses.

    Args:
        config: Master analysis configuration.
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        data: torch.Tensor,
        algebra: Optional[AlgebraLike] = None,
    ) -> AnalysisReport:
        """Run the full geometric analysis pipeline.

        Args:
            data: Raw ``[N, D]`` or pre-embedded ``[N, C, 2^n]`` tensor.
            algebra: Required when *data* is pre-embedded.  Optional
                when raw -- will be created from the signature estimate.

        Returns:
            :class:`AnalysisReport`.
        """
        cfg = self.config
        report = AnalysisReport()
        t0 = time.time()

        data = as_analysis_tensor(data, device=cfg.device, dtype=cfg.dtype)
        report.metadata["data_shape"] = list(data.shape)
        report.metadata["config_device"] = cfg.device

        if data.ndim == 3 and algebra is not None:
            # Pre-embedded mode
            report = self._run_ga_analyses(data, algebra, report)
        elif data.ndim == 2 and algebra is not None:
            # Raw + known algebra
            mv_data = self._embed_raw(data, algebra)
            report = self._run_ga_analyses(mv_data, algebra, report)
        elif data.ndim == 2:
            # Full raw mode
            report = self._run_full_pipeline(data, report)
        else:
            raise ValueError(f"Unexpected data shape {data.shape}. Expected [N, D] or [N, C, dim].")

        report.metadata["elapsed_seconds"] = round(time.time() - t0, 2)
        return report

    def _run_full_pipeline(self, data: torch.Tensor, report: AnalysisReport) -> AnalysisReport:
        cfg = self.config

        # Sampling
        sampled, sample_meta = StatisticalSampler.sample(data, cfg.sampling)
        if isinstance(sampled, list):
            # bootstrap returns list -- use first for pipeline, rest for CI
            sampled = sampled[0]
        report.metadata["sampling"] = sample_meta

        # Dimension analysis
        dim_result = None
        if cfg.run_dimension:
            dimension_analyzer = CovarianceDimensionAnalyzer(
                device=cfg.device,
                dtype=cfg.dtype,
                energy_threshold=cfg.energy_threshold,
            )
            dim_result = dimension_analyzer.analyze(sampled)
            report.dimension = dim_result

        # Experimental signature estimation
        sig_result = None
        if cfg.run_signature_estimation:
            signature_analyzer = SignatureProbeAnalyzer(device=cfg.device, dtype=cfg.dtype)
            sig_result = signature_analyzer.analyze(sampled, dim_result=dim_result)
            report.signature_estimate = sig_result

        # Create algebra and embed (n >= 2 for meaningful GA structure)
        if sig_result is not None:
            p, q, r = sig_result.estimated_signature
        elif dim_result is not None:
            p, q, r = dim_result.broken_stick_dimension, 0, 0
        else:
            p, q, r = min(sampled.shape[1], CONSTANTS.pipeline_fallback_dim_cap), 0, 0
        # Ensure n >= 2 so grade-2 (bivectors) exist for GA analyses
        if p + q + r < CONSTANTS.pipeline_min_ga_n:
            p = max(p, CONSTANTS.pipeline_min_ga_n - q - r)

        algebra = make_algebra(p, q, r, device=cfg.device, dtype=cfg.dtype, default_grades=(1,))
        mv_data = self._embed_raw(sampled, algebra)

        # GA analyses
        report = self._run_ga_analyses(mv_data, algebra, report)

        return report

    def _run_ga_analyses(
        self,
        mv_data: torch.Tensor,
        algebra: AlgebraLike,
        report: AnalysisReport,
    ) -> AnalysisReport:
        cfg = self.config

        tasks = {}
        if cfg.run_spectral:
            tasks["spectral"] = lambda: SpectralAnalyzer(algebra).analyze(mv_data)
        if cfg.run_transformation_diagnostics:
            tasks["transformation"] = lambda: TransformationDiagnosticsAnalyzer(
                algebra, low_energy_threshold=cfg.energy_threshold
            ).analyze(mv_data)
        if cfg.run_commutator:
            tasks["commutator"] = lambda: CommutatorAnalyzer(algebra).analyze(mv_data)

        if len(tasks) <= 1:
            # Sequential
            for name, fn in tasks.items():
                setattr(report, name, fn())
        else:
            # Parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=CONSTANTS.pipeline_parallel_workers) as pool:
                futures = {name: pool.submit(fn) for name, fn in tasks.items()}
                for name, fut in futures.items():
                    setattr(report, name, fut.result())

        # Recount near-commuting modes from the adjoint diagnostics when available.
        if (
            cfg.run_transformation_diagnostics
            and cfg.run_commutator
            and report.transformation is not None
            and report.commutator is not None
        ):
            if mv_data.ndim == 3:
                flat = mv_data.mean(dim=1)
            else:
                flat = mv_data
            transformation_analyzer = TransformationDiagnosticsAnalyzer(
                algebra, low_energy_threshold=cfg.energy_threshold
            )
            near_commuting_count, commuting_skipped = transformation_analyzer._near_commuting_modes_with_skips(
                flat,
                commutator_result=report.commutator,
            )
            report.transformation.near_commuting_mode_count = near_commuting_count
            report.transformation.skipped.pop("near_commuting_modes", None)
            report.transformation.skipped.update(commuting_skipped)

        return report

    @staticmethod
    def _embed_raw(data: torch.Tensor, algebra: AlgebraLike) -> torch.Tensor:
        """Embed raw ``[N, D]`` data as grade-1 multivectors ``[N, 1, dim]``."""
        n = algebra.n
        D = data.shape[1]
        if D > n:
            data = data[:, :n]
        elif D < n:
            pad = torch.zeros(data.shape[0], n - D, device=data.device, dtype=data.dtype)
            data = torch.cat([data, pad], dim=-1)
        mv = algebra.embed_vector(data.to(device=algebra.device, dtype=algebra.dtype))  # [N, dim]
        return mv.unsqueeze(1)  # [N, 1, dim]
