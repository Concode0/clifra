# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Public records for geometric diagnostics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from clifra.core.planning.policy import FULL_TABLE_AUTO_MAX_N, FULL_TABLE_EXPLICIT_MAX_N

GP_SPECTRUM_MATRIX_ENTRIES = 1 << (2 * 10)
ADJOINT_MATRIX_ENTRIES = 1 << (2 * 8)
AUTO_FULL_PRODUCT_PAIRS = 1 << (2 * FULL_TABLE_AUTO_MAX_N)
EXPLICIT_ACTION_MATRIX_LANES = 1 << FULL_TABLE_EXPLICIT_MAX_N
EXPLICIT_ACTION_MATRIX_ENTRIES = EXPLICIT_ACTION_MATRIX_LANES * EXPLICIT_ACTION_MATRIX_LANES


@dataclass
class AnalysisConstants:
    """Central registry of tuneable constants for the analysis toolkit.

    Fixed thresholds and budgets that affect analytical decisions are grouped
    here so they can be adjusted in one place. Numerical guards belong in
    :mod:`clifra.core.foundation.numerics`.

    Attributes:
        alignment_report_max_dissimilarity: Maximum neighborhood-connection
            dissimilarity accepted by the alignment report.
            See :meth:`NeighborhoodBivectorFlow.alignment_threshold_report`
            and :meth:`CoordinateLiftAnalyzer.compare_lifts`.
        bv_sq_elliptic_bound: Bivector square value below which the
            bivector is classified as *elliptic* (negative definite)
            in signature analysis.
        bv_sq_hyperbolic_bound: Bivector square value above which the
            bivector is classified as *hyperbolic* (positive definite)
            in signature analysis.
        near_commuting_mode_threshold: Normalized commutator norm below
            which a mode is counted as near-commuting.
        basis_reflection_score_threshold: Maximum basis-reflection
            distribution score counted in the report summary.
        gp_spectrum_matrix_entries: Maximum square-matrix entries for
            the full geometric-product operator spectrum.
        adjoint_matrix_entries: Maximum square-matrix entries for the
            adjoint-operator eigensolver.
        analysis_product_pairs: Maximum planned product interactions for
            optional full-layout analysis subroutines.
        gp_spectrum_n_samples: Number of data samples used when building
            the GP left-multiplication matrix.
    """

    alignment_report_max_dissimilarity: float = 0.5
    bv_sq_elliptic_bound: float = -0.5
    bv_sq_hyperbolic_bound: float = 0.5
    near_commuting_mode_threshold: float = 0.05
    basis_reflection_score_threshold: float = 0.1
    gp_spectrum_matrix_entries: int = GP_SPECTRUM_MATRIX_ENTRIES
    gp_spectrum_product_pairs: int = GP_SPECTRUM_MATRIX_ENTRIES
    adjoint_matrix_entries: int = ADJOINT_MATRIX_ENTRIES
    analysis_product_pairs: int = AUTO_FULL_PRODUCT_PAIRS
    reflection_product_pairs: int = AUTO_FULL_PRODUCT_PAIRS
    near_commuting_mode_product_pairs: int = AUTO_FULL_PRODUCT_PAIRS
    signature_probe_action_matrix_entries: int = EXPLICIT_ACTION_MATRIX_ENTRIES
    signature_probe_action_matrix_lanes: int = EXPLICIT_ACTION_MATRIX_LANES
    gp_spectrum_n_samples: int = 50
    default_k_neighbors: int = 8
    default_energy_threshold: float = 0.05
    default_dtype: torch.dtype = torch.float32
    dimension_k_local: int = 20
    dimension_local_max_samples: int = 5000
    dimension_min_reported_dimension: int = 1
    dimension_lift_positive_fill: float = 1.0
    dimension_lift_negative_square_fill: float = 0.0
    bivector_interpolation_steps: int = 10
    sampling_max_samples: int = 500
    sampling_seed: int = 42
    sampling_bootstrap_resamples: int = 100
    sampling_recommended_min_samples: int = 500
    sampling_recommended_feature_multiplier: int = 20
    stratified_points_per_stratum: int = 20
    stratified_min_strata: int = 2
    stratified_max_strata: int = 10
    stratified_algebra_max_dim: int = 6
    signature_probe_num_probes: int = 6
    signature_probe_epochs: int = 80
    signature_probe_lr: float = 0.005
    signature_probe_channels: int = 4
    signature_probe_connection_dissimilarity_weight: float = 0.3
    signature_probe_sparsity_weight: float = 0.01
    signature_probe_parallel_worker_cap: int = 4
    signature_probe_cga_extra_dims: int = 2
    signature_probe_rotor_init_std: float = 0.01
    signature_probe_bias_minor_weight: float = 0.1
    signature_probe_bias_noise_std: float = 0.05
    signature_probe_projective_init_bound: float = 0.5
    signature_probe_random_init_std: float = 0.3
    signature_search_max_dim: int = FULL_TABLE_EXPLICIT_MAX_N - 2
    signature_bootstrap_resamples: int = 10
    signature_bootstrap_max_samples: int = 500
    low_energy_vector_threshold: float = 0.01
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
        run_dimension: Enable covariance-dimension diagnostics.
        run_signature_estimation: Enable rotor-probe signature estimation.
        run_spectral: Enable spectral analysis.
        run_transformation_diagnostics: Enable operational transformation diagnostics.
        run_commutator: Enable commutator analysis.
        energy_threshold: Cutoff for "active" components (shared across
            analyzers).
        k_neighbors: Number of nearest neighbors for local analyses.
        verbose: Print progress messages.
    """

    device: str = "cpu"
    dtype: torch.dtype = CONSTANTS.default_dtype
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    run_dimension: bool = True
    run_signature_estimation: bool = True
    run_spectral: bool = True
    run_transformation_diagnostics: bool = True
    run_commutator: bool = True
    energy_threshold: float = CONSTANTS.default_energy_threshold
    k_neighbors: int = CONSTANTS.default_k_neighbors
    verbose: bool = False


@dataclass
class DimensionResult:
    """Output of :class:`CovarianceDimensionAnalyzer`.

    Attributes:
        broken_stick_dimension: Broken-stick component count, clamped to at
            least one for downstream algebra construction.
        participation_ratio: Covariance-spectrum participation ratio
            ``(Sum_lam)^2 / Sum_lam^2``.
        eigenvalues: Covariance eigenvalues sorted descending.
        broken_stick_count: Number of components exceeding
            the broken-stick null model.
        explained_variance_ratio: Per-component explained variance ratio.
        local_participation_ratios: Per-point neighborhood covariance
            participation ratios (optional).
    """

    broken_stick_dimension: int
    participation_ratio: float
    eigenvalues: torch.Tensor
    broken_stick_count: int
    explained_variance_ratio: torch.Tensor
    local_participation_ratios: Optional[torch.Tensor] = None


@dataclass
class SignatureEstimate:
    """Output of :class:`SignatureProbeAnalyzer`.

    Attributes:
        estimated_signature: Probe-selected ``(p, q, r)`` candidate from the
            learned ranking heuristic.
        connection_alignment: Best probe's mean absolute connection alignment.
        connection_dissimilarity: Best probe's neighboring-connection
            dissimilarity.
        energy_breakdown: Per-bivector energy dict from
            ``RotorProbeSignatureEstimator._analyze_bivector_energy``.
        input_dimension_used: Reduced input dimension actually searched
            (``None`` if no reduction was applied).
    """

    estimated_signature: Tuple[int, int, int]
    connection_alignment: float
    connection_dissimilarity: float
    energy_breakdown: Dict
    input_dimension_used: Optional[int] = None


@dataclass
class SpectralResult:
    """Output of :class:`SpectralAnalyzer`.

    Attributes:
        grade_energy: Mean positive lane grade energy ``[n+1]``.
        mean_bivector_norm: Norm summary of the mean bivector field.
        mean_bivector_components: Representative full-layout bivector tensors.
        gp_action_eigenvalue_magnitudes: Eigenvalues of the geometric-product left-action
            operator (``None`` if the algebra was too large).
        skipped: Optional analysis subreports skipped by feasibility policy.
    """

    grade_energy: torch.Tensor
    mean_bivector_norm: torch.Tensor
    mean_bivector_components: List[torch.Tensor]
    gp_action_eigenvalue_magnitudes: Optional[torch.Tensor] = None
    skipped: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class TransformationDiagnosticsResult:
    """Operational output of :class:`TransformationDiagnosticsAnalyzer`.

    Attributes:
        low_energy_vector_directions: Indices whose normalized observed
            grade-1 coefficient energy is below the configured threshold.
        normalized_vector_energy: Per-basis-vector normalized energy ``[n]``.
        odd_grade_energy_fraction: Fraction of energy in odd-grade components,
            in ``[0, 1]``.  0 = data lives entirely in the even
            sub-algebra; 1 = entirely odd grades.
        basis_reflection_scores: List of dicts, each with ``direction`` index
            and coefficient-distribution distance ``score`` (lower is closer).
        near_commuting_mode_count: Number of modes below the configured
            normalized commutator threshold.
        skipped: Optional subreports skipped by feasibility policy.
    """

    low_energy_vector_directions: List[int]
    normalized_vector_energy: torch.Tensor
    odd_grade_energy_fraction: float
    basis_reflection_scores: List[Dict]
    near_commuting_mode_count: int
    skipped: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class CommutatorResult:
    """Output of :class:`CommutatorAnalyzer`.

    Attributes:
        pairwise_commutator_norms: ``[D, D]`` mean pairwise commutator norms.
        adjoint_eigenvalue_magnitudes: Magnitudes of eigenvalues of ``ad_mu``.
        mean_commutator_norm: Scalar summary ``E[||[x_i, mu]||_2]``.
        bivector_bracket_closure: Dict with ``structure_constants`` ``[k, k, k]``
            tensor, ``closure_error`` scalar, and ``basis_indices`` list.
        skipped: Optional commutator subreports skipped by feasibility policy.
    """

    pairwise_commutator_norms: torch.Tensor
    adjoint_eigenvalue_magnitudes: torch.Tensor
    mean_commutator_norm: float
    bivector_bracket_closure: Dict
    skipped: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Full analysis report combining all sub-analyzers.

    Attributes:
        dimension: Covariance-spectrum dimension diagnostics.
        signature_estimate: Rotor-probe signature estimate.
        spectral: Spectral analysis results.
        transformation: Operational transformation diagnostics.
        commutator: Commutator analysis results.
        metadata: Timing, configuration, and data-shape information.
    """

    dimension: Optional[DimensionResult] = None
    signature_estimate: Optional[SignatureEstimate] = None
    spectral: Optional[SpectralResult] = None
    transformation: Optional[TransformationDiagnosticsResult] = None
    commutator: Optional[CommutatorResult] = None
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = ["=== Geometric Analysis Report ==="]

        if self.dimension is not None:
            d = self.dimension
            lines.append(f"\n[Dimension]")
            lines.append(f"  Broken-stick dimension:       {d.broken_stick_dimension}")
            lines.append(f"  Participation ratio:          {d.participation_ratio:.2f}")
            top_k = min(5, len(d.eigenvalues))
            ev = ", ".join(f"{v:.4f}" for v in d.eigenvalues[:top_k].tolist())
            lines.append(f"  Top eigenvalues:              [{ev}]")

        if self.signature_estimate is not None:
            s = self.signature_estimate
            p, q, r = s.estimated_signature
            lines.append(f"\n[Signature estimate]")
            lines.append(f"  Candidate Cl({p},{q},{r})")
            lines.append(
                f"  Connection alignment: {s.connection_alignment:.3f}  dissimilarity: {s.connection_dissimilarity:.3f}"
            )
            if s.input_dimension_used is not None:
                lines.append(f"  (searched in {s.input_dimension_used}D reduced space)")

        if self.spectral is not None:
            sp = self.spectral
            lines.append(f"\n[Spectral]")
            ge = ", ".join(f"{v:.4f}" for v in sp.grade_energy.tolist())
            lines.append(f"  Grade energy: [{ge}]")
            bv = ", ".join(f"{v:.4f}" for v in sp.mean_bivector_norm.tolist())
            lines.append(f"  Mean bivector norm: [{bv}]")
            if sp.gp_action_eigenvalue_magnitudes is not None:
                top = min(5, len(sp.gp_action_eigenvalue_magnitudes))
                gpe = ", ".join(f"{v:.4f}" for v in sp.gp_action_eigenvalue_magnitudes[:top].tolist())
                lines.append(f"  GP action eigenvalue magnitudes (top {top}): [{gpe}]")
            if sp.skipped:
                lines.append(f"  Skipped: {', '.join(sorted(sp.skipped))}")

        if self.transformation is not None:
            tr = self.transformation
            lines.append(f"\n[Transformation diagnostics]")
            lines.append(f"  Low-energy vector directions: {tr.low_energy_vector_directions}")
            lines.append(f"  Odd-grade energy fraction: {tr.odd_grade_energy_fraction:.4f}")
            lines.append(f"  Near-commuting mode count: {tr.near_commuting_mode_count}")
            close_reflections = sum(
                1 for item in tr.basis_reflection_scores if item["score"] < CONSTANTS.basis_reflection_score_threshold
            )
            lines.append(f"  Low-distance basis reflections: {close_reflections}")
            if tr.skipped:
                lines.append(f"  Skipped: {', '.join(sorted(tr.skipped))}")

        if self.commutator is not None:
            c = self.commutator
            lines.append(f"\n[Commutator]")
            lines.append(f"  Mean commutator norm: {c.mean_commutator_norm:.4f}")
            top = min(5, len(c.adjoint_eigenvalue_magnitudes))
            es = ", ".join(f"{v:.4f}" for v in c.adjoint_eigenvalue_magnitudes[:top].tolist())
            lines.append(f"  Adjoint eigenvalue magnitudes (top {top}): [{es}]")
            ce = c.bivector_bracket_closure.get("closure_error", None)
            if ce is not None:
                lines.append(f"  Lie bracket closure error: {ce:.4f}")
            if c.skipped:
                lines.append(f"  Skipped: {', '.join(sorted(c.skipped))}")

        if self.metadata:
            lines.append(f"\n[Metadata]")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        return "\n".join(lines)
