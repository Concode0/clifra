# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Shared dataclass types for the geometric analysis toolkit."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from clifra.core.planning.policy import DENSE_AUTO_MAX_N, DENSE_EXPLICIT_MAX_N


@dataclass
class AnalysisConstants:
    """Central registry of tuneable constants for the analysis toolkit.

    All magic numbers that affect analytical decisions live here so they
    can be adjusted in one place. Numerical guards belong in
    :mod:`clifra.core.foundation.numerics`.

    Attributes:
        curvature_causal_threshold: Maximum curvature score for a flow
            field to be classified as "causal" (used in
            :meth:`GeodesicFlow.causal_report` and
            :meth:`DimensionLifter.test`).
        bv_sq_elliptic_bound: Bivector square value below which the
            bivector is classified as *elliptic* (negative definite)
            in signature analysis.
        bv_sq_hyperbolic_bound: Bivector square value above which the
            bivector is classified as *hyperbolic* (positive definite)
            in signature analysis.
        continuous_symmetry_threshold: Normalised commutator norm below
            which a bivector is treated as a continuous-symmetry
            generator.
        reflection_score_threshold: Maximum reflection score for a
            direction to be counted as a reflection symmetry in the
            report summary.
        gp_spectrum_max_n: Maximum algebra dimension *n* for which the
            full geometric-product operator spectrum is computed
            (``dim = 2^n``).
        adjoint_max_n: Maximum algebra dimension *n* for which the
            adjoint-operator matrix is materialised.
        gp_spectrum_n_samples: Number of data samples used when building
            the GP left-multiplication matrix.
    """

    curvature_causal_threshold: float = 0.5
    bv_sq_elliptic_bound: float = -0.5
    bv_sq_hyperbolic_bound: float = 0.5
    continuous_symmetry_threshold: float = 0.05
    reflection_score_threshold: float = 0.1
    gp_spectrum_max_n: int = 10
    adjoint_max_n: int = 8
    gp_spectrum_n_samples: int = 50
    dense_analysis_max_n: int = DENSE_AUTO_MAX_N
    reflection_analysis_max_n: int = DENSE_AUTO_MAX_N
    default_k_neighbors: int = 8
    default_energy_threshold: float = 0.05
    default_dtype: torch.dtype = torch.float32
    dimension_k_local: int = 20
    dimension_local_max_samples: int = 5000
    dimension_min_intrinsic_dim: int = 1
    dimension_lift_positive_fill: float = 1.0
    dimension_lift_null_fill: float = 0.0
    geodesic_interpolation_steps: int = 10
    sampling_max_samples: int = 500
    sampling_seed: int = 42
    sampling_bootstrap_resamples: int = 100
    sampling_recommended_min_samples: int = 500
    sampling_recommended_feature_multiplier: int = 20
    stratified_points_per_stratum: int = 20
    stratified_min_strata: int = 2
    stratified_max_strata: int = 10
    stratified_algebra_max_dim: int = 6
    metric_search_num_probes: int = 6
    metric_search_probe_epochs: int = 80
    metric_search_probe_lr: float = 0.005
    metric_search_probe_channels: int = 4
    metric_search_curvature_weight: float = 0.3
    metric_search_sparsity_weight: float = 0.01
    metric_search_parallel_worker_cap: int = 4
    metric_search_dense_max_n: int = DENSE_EXPLICIT_MAX_N
    metric_search_cga_extra_dims: int = 2
    metric_search_rotor_init_std: float = 0.01
    metric_search_bias_minor_weight: float = 0.1
    metric_search_bias_noise_std: float = 0.05
    metric_search_projective_init_bound: float = 0.5
    metric_search_random_init_std: float = 0.3
    signature_search_max_dim: int = DENSE_EXPLICIT_MAX_N - 2
    signature_bootstrap_resamples: int = 10
    signature_bootstrap_max_samples: int = 500
    spectral_max_simple_components: int = 5
    symmetry_null_threshold: float = 0.01
    commutator_max_bivectors: int = 15
    pipeline_parallel_workers: int = 3
    pipeline_fallback_dim_cap: int = 6
    pipeline_min_ga_n: int = 2


# Module-level singleton -- importable as ``from ._types import CONSTANTS``.
CONSTANTS = AnalysisConstants()


@dataclass
class SamplingConfig:
    """Controls how raw data is sampled before analysis.

    Attributes:
        strategy: One of ``"random"``, ``"stratified"``, ``"bootstrap"``,
            ``"passthrough"``.
        max_samples: Maximum number of samples to draw (ignored for
            ``"passthrough"``).
        seed: Random seed for reproducibility.
        n_bootstrap: Number of bootstrap resamples (only for ``"bootstrap"``).
        n_strata: Number of clusters for ``"stratified"`` sampling.
            Auto-determined when *None*.
    """

    strategy: str = "random"
    max_samples: int = CONSTANTS.sampling_max_samples
    seed: int = CONSTANTS.sampling_seed
    n_bootstrap: int = CONSTANTS.sampling_bootstrap_resamples
    n_strata: Optional[int] = None


@dataclass
class AnalysisConfig:
    """Master configuration for the full analysis pipeline.

    Attributes:
        device: Torch device string.
        sampling: Sampling configuration.
        run_dimension: Enable effective-dimension analysis.
        run_signature: Enable metric-signature search.
        run_spectral: Enable spectral analysis.
        run_symmetry: Enable symmetry / null detection.
        run_commutator: Enable commutator analysis.
        energy_threshold: Cutoff for "active" components (shared across
            analyzers).
        k_neighbors: Number of nearest neighbours for local analyses.
        verbose: Print progress messages.
    """

    device: str = "cpu"
    dtype: torch.dtype = CONSTANTS.default_dtype
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    run_dimension: bool = True
    run_signature: bool = True
    run_spectral: bool = True
    run_symmetry: bool = True
    run_commutator: bool = True
    energy_threshold: float = CONSTANTS.default_energy_threshold
    k_neighbors: int = CONSTANTS.default_k_neighbors
    verbose: bool = False


@dataclass
class DimensionResult:
    """Output of :class:`EffectiveDimensionAnalyzer`.

    Attributes:
        intrinsic_dim: Estimated integer intrinsic dimensionality (via
            broken-stick).
        participation_ratio: Smooth dimensionality estimate
            ``(Sum_lam)^2 / Sum_lam^2``.
        eigenvalues: Covariance eigenvalues sorted descending.
        broken_stick_threshold: Number of significant components exceeding
            the broken-stick null model.
        explained_variance_ratio: Per-component explained variance ratio.
        local_dims: Per-point local dimension estimates (optional).
    """

    intrinsic_dim: int
    participation_ratio: float
    eigenvalues: torch.Tensor
    broken_stick_threshold: int
    explained_variance_ratio: torch.Tensor
    local_dims: Optional[torch.Tensor] = None


@dataclass
class SignatureResult:
    """Output of :class:`SignatureSearchAnalyzer`.

    Attributes:
        signature: Discovered ``(p, q, r)`` metric signature.
        coherence: Best-probe geodesic-flow coherence.
        curvature: Best-probe geodesic-flow curvature.
        energy_breakdown: Per-bivector energy dict from
            ``MetricSearch._analyze_bivector_energy``.
        effective_dim_used: Reduced dimension that was actually searched
            (``None`` if no reduction was applied).
    """

    signature: Tuple[int, int, int]
    coherence: float
    curvature: float
    energy_breakdown: Dict
    effective_dim_used: Optional[int] = None


@dataclass
class SpectralResult:
    """Output of :class:`SpectralAnalyzer`.

    Attributes:
        grade_energy: Mean Hermitian grade energy ``[n+1]``.
        bivector_spectrum: Singular values of decomposed bivector field.
        simple_components: List of simple-bivector tensors from
            decomposition.
        gp_eigenvalues: Eigenvalues of the geometric-product left-action
            operator (``None`` if the algebra was too large).
    """

    grade_energy: torch.Tensor
    bivector_spectrum: torch.Tensor
    simple_components: List[torch.Tensor]
    gp_eigenvalues: Optional[torch.Tensor] = None


@dataclass
class SymmetryResult:
    """Output of :class:`SymmetryDetector`.

    Attributes:
        null_directions: Indices of near-null basis vectors.
        null_scores: Per-basis-vector null score ``[n]``.
        involution_symmetry: Fraction of energy in odd-grade components,
            in ``[0, 1]``.  0 = data lives entirely in the even
            sub-algebra; 1 = entirely odd grades.
        reflection_symmetries: List of dicts, each with ``direction`` index
            and ``score`` (lower = more symmetric).
        continuous_symmetry_dim: Dimension of the detected continuous
            symmetry group.
    """

    null_directions: List[int]
    null_scores: torch.Tensor
    involution_symmetry: float
    reflection_symmetries: List[Dict]
    continuous_symmetry_dim: int


@dataclass
class CommutatorResult:
    """Output of :class:`CommutatorAnalyzer`.

    Attributes:
        commutativity_matrix: ``[D, D]`` pairwise commutativity indices.
        exchange_spectrum: Eigenvalues of the adjoint operator ``ad_mu``.
        mean_commutator_norm: Scalar summary ``E[||[x_i, mu]||_2]``.
        lie_bracket_structure: Dict with ``structure_constants`` ``[k, k, k]``
            tensor, ``closure_error`` scalar, and ``basis_indices`` list.
    """

    commutativity_matrix: torch.Tensor
    exchange_spectrum: torch.Tensor
    mean_commutator_norm: float
    lie_bracket_structure: Dict


@dataclass
class AnalysisReport:
    """Full analysis report combining all sub-analyzers.

    Attributes:
        dimension: Effective-dimension results.
        signature: Metric-signature search results.
        spectral: Spectral analysis results.
        symmetry: Symmetry / null detection results.
        commutator: Commutator analysis results.
        metadata: Timing, configuration, and data-shape information.
    """

    dimension: Optional[DimensionResult] = None
    signature: Optional[SignatureResult] = None
    spectral: Optional[SpectralResult] = None
    symmetry: Optional[SymmetryResult] = None
    commutator: Optional[CommutatorResult] = None
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = ["=== Geometric Analysis Report ==="]

        if self.dimension is not None:
            d = self.dimension
            lines.append(f"\n[Dimension]")
            lines.append(f"  Intrinsic dim (broken-stick): {d.intrinsic_dim}")
            lines.append(f"  Participation ratio:          {d.participation_ratio:.2f}")
            top_k = min(5, len(d.eigenvalues))
            ev = ", ".join(f"{v:.4f}" for v in d.eigenvalues[:top_k].tolist())
            lines.append(f"  Top eigenvalues:              [{ev}]")

        if self.signature is not None:
            s = self.signature
            p, q, r = s.signature
            lines.append(f"\n[Signature]")
            lines.append(f"  Cl({p},{q},{r})")
            lines.append(f"  Coherence: {s.coherence:.3f}  Curvature: {s.curvature:.3f}")
            if s.effective_dim_used is not None:
                lines.append(f"  (searched in {s.effective_dim_used}D reduced space)")

        if self.spectral is not None:
            sp = self.spectral
            lines.append(f"\n[Spectral]")
            ge = ", ".join(f"{v:.4f}" for v in sp.grade_energy.tolist())
            lines.append(f"  Grade energy: [{ge}]")
            bv = ", ".join(f"{v:.4f}" for v in sp.bivector_spectrum.tolist())
            lines.append(f"  Bivector spectrum: [{bv}]")
            if sp.gp_eigenvalues is not None:
                top = min(5, len(sp.gp_eigenvalues))
                gpe = ", ".join(f"{v:.4f}" for v in sp.gp_eigenvalues[:top].tolist())
                lines.append(f"  GP eigenvalues (top {top}): [{gpe}]")

        if self.symmetry is not None:
            sy = self.symmetry
            lines.append(f"\n[Symmetry]")
            lines.append(f"  Null directions: {sy.null_directions}")
            lines.append(f"  Involution symmetry: {sy.involution_symmetry:.4f}")
            lines.append(f"  Continuous symmetry dim: {sy.continuous_symmetry_dim}")
            n_refl = sum(1 for r in sy.reflection_symmetries if r["score"] < CONSTANTS.reflection_score_threshold)
            lines.append(f"  Reflection symmetries: {n_refl} detected")

        if self.commutator is not None:
            c = self.commutator
            lines.append(f"\n[Commutator]")
            lines.append(f"  Mean commutator norm: {c.mean_commutator_norm:.4f}")
            top = min(5, len(c.exchange_spectrum))
            es = ", ".join(f"{v:.4f}" for v in c.exchange_spectrum[:top].tolist())
            lines.append(f"  Exchange spectrum (top {top}): [{es}]")
            ce = c.lie_bracket_structure.get("closure_error", None)
            if ce is not None:
                lines.append(f"  Lie bracket closure error: {ce:.4f}")

        if self.metadata:
            lines.append(f"\n[Metadata]")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)
