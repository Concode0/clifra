# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Effective-dimension analysis and dimension lifting.

Implements participation ratio, broken-stick null model, local-dimension
estimation, and algebraic dimension lifting to determine how many
dimensions the data actually occupies and whether lifting reveals
latent structure.
"""

from typing import Dict

import torch

from clifra.core.config import make_algebra
from clifra.core.foundation.module import AlgebraLike
from clifra.core.foundation.numerics import DEFAULT_EPS_MULTIPLIER, eps_for

from ._types import CONSTANTS, DimensionResult
from ._utils import analysis_dtype, as_analysis_tensor


class EffectiveDimensionAnalyzer:
    """Estimate effective intrinsic dimensionality of data.

    Uses the covariance eigenvalue spectrum together with the
    participation ratio (random matrix theory) and the broken-stick
    model (MacArthur 1957) for significance testing.

    Args:
        device: Torch device string.
        dtype: Floating-point dtype used for covariance computations.
            Defaults to ``torch.float32``.  Pass ``torch.float64`` for
            higher-precision analyses.
        k_local: Number of neighbors for local-dimension estimation.
        energy_threshold: Minimum normalized eigenvalue to count as
            active.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = CONSTANTS.default_dtype,
        k_local: int = CONSTANTS.dimension_k_local,
        energy_threshold: float = CONSTANTS.default_energy_threshold,
    ):
        self.device = device
        self.dtype = analysis_dtype(dtype)
        self.k_local = k_local
        self.energy_threshold = energy_threshold
        self._eps: float = eps_for(self.dtype, multiplier=DEFAULT_EPS_MULTIPLIER)
        self._eps_sq: float = self._eps**2

    def analyze(self, data: torch.Tensor) -> DimensionResult:
        """Full effective-dimension analysis.

        Args:
            data: ``[N, D]`` raw data tensor.

        Returns:
            :class:`DimensionResult` with eigenvalues, participation
            ratio, broken-stick threshold, and optional local dims.
        """
        data = as_analysis_tensor(data, device=self.device, dtype=self.dtype)
        if data.ndim != 2:
            raise ValueError(f"data must have shape [N, D], got {tuple(data.shape)}")
        N, D = data.shape

        eigenvalues = self._covariance_eigenvalues(data)  # descending
        pr = self._participation_ratio(eigenvalues)

        total = eigenvalues.sum()
        if total > 0:
            evr = eigenvalues / total
        else:
            evr = torch.zeros_like(eigenvalues)

        bs_threshold = self._broken_stick_threshold(eigenvalues, D)

        local_dims = None
        if N < CONSTANTS.dimension_local_max_samples and N > self.k_local + 1:
            local_dims = self._local_dimensions(data, self.k_local)

        intrinsic_dim = max(CONSTANTS.dimension_min_intrinsic_dim, bs_threshold)

        return DimensionResult(
            intrinsic_dim=intrinsic_dim,
            participation_ratio=pr,
            eigenvalues=eigenvalues,
            broken_stick_threshold=bs_threshold,
            explained_variance_ratio=evr,
            local_dims=local_dims,
        )

    def reduce(self, data: torch.Tensor, target_dim: int) -> torch.Tensor:
        """PCA projection to *target_dim* dimensions.

        Args:
            data: ``[N, D]`` tensor.
            target_dim: Target dimensionality.

        Returns:
            ``[N, target_dim]`` projected tensor.
        """
        data = as_analysis_tensor(data, device=self.device, dtype=self.dtype)
        if data.ndim != 2:
            raise ValueError(f"data must have shape [N, D], got {tuple(data.shape)}")
        if target_dim <= 0:
            raise ValueError(f"target_dim must be positive, got {target_dim}")
        target_dim = min(int(target_dim), data.shape[1])
        mean = data.mean(dim=0, keepdim=True)
        centered = data - mean
        # Economy SVD
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        return U[:, :target_dim] * S[:target_dim].unsqueeze(0)

    @staticmethod
    def _covariance_eigenvalues(data: torch.Tensor) -> torch.Tensor:
        """Eigenvalues of the sample covariance, sorted descending."""
        N = data.shape[0]
        mean = data.mean(dim=0, keepdim=True)
        centered = data - mean
        cov = (centered.T @ centered) / max(N - 1, 1)
        # eigh returns ascending; flip
        eigvals = torch.linalg.eigvalsh(cov)
        return eigvals.flip(0).clamp(min=0.0)

    def _participation_ratio(self, eigenvalues: torch.Tensor) -> float:
        """``(Sum lam)^2 / Sum lam^2`` -- smooth dimensionality estimator."""
        s1 = eigenvalues.sum()
        s2 = (eigenvalues**2).sum()
        if s2 < self._eps_sq:
            return 0.0
        return (s1**2 / s2).item()

    @staticmethod
    def _broken_stick(d: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Expected eigenvalues under the broken-stick null model.

        The *k*-th expected value is ``(1/d) * Sum_{j=k+1}^{d} 1/j``
        (0-indexed).
        """
        inv = 1.0 / torch.arange(1, d + 1, dtype=torch.float64)
        # Reverse cumulative sum gives Sum_{j=k+1}^{d} 1/j for 0-indexed k
        expected = inv.flip(0).cumsum(0).flip(0) / d
        return expected.to(dtype=dtype)

    def _broken_stick_threshold(self, eigenvalues: torch.Tensor, d: int) -> int:
        """Number of components exceeding the broken-stick null."""
        if d <= 0:
            return 0
        expected = self._broken_stick(d, dtype=eigenvalues.dtype).to(eigenvalues.device)
        total = eigenvalues.sum()
        if total < self._eps_sq:
            return 0
        normed = eigenvalues[:d] / total
        return int((normed > expected[: len(normed)]).sum().item())

    def _local_dimensions(self, data: torch.Tensor, k: int) -> torch.Tensor:
        """Per-point local dimension via neighborhood PCA."""
        dists = torch.cdist(data, data)  # [N, N]
        # k+1 because the closest point is itself
        _, knn_idx = dists.topk(k + 1, largest=False, dim=1)
        knn_idx = knn_idx[:, 1:]  # drop self

        # Batched neighborhood PCA
        all_nbrs = data[knn_idx]  # [N, k, D]
        centered = all_nbrs - all_nbrs.mean(dim=1, keepdim=True)
        k_count = centered.shape[1]
        if k_count < 2:
            return data.new_zeros(data.shape[0])
        cov = torch.bmm(centered.transpose(1, 2), centered) / max(k_count - 1, 1)
        eigvals = torch.linalg.eigvalsh(cov).flip(-1).clamp(min=0.0)  # [N, D] desc

        # Vectorized participation ratio
        s1 = eigvals.sum(dim=-1)
        s2 = (eigvals**2).sum(dim=-1)
        local_dims = torch.where(s2 > self._eps_sq, s1**2 / s2, torch.zeros_like(s1))

        return local_dims


class DimensionLifter:
    """Tests whether lifting data to a higher-dimensional algebra reveals structure.

    The hypothesis: data living on an n-dimensional manifold may possess
    latent structure that only becomes visible in Cl(n+1, q) or Cl(n, q+1).

    Lifting appends extra coordinates to the grade-1 embedding:

    - **Positive lift** ``Cl(p, q) -> Cl(p+1, q)``: adds a spacelike dimension.
      The extra coordinate is set to 1 (projective / homogeneous lift).
    - **Null lift** ``Cl(p, q) -> Cl(p, q+1)``: adds a timelike dimension.
      The extra coordinate is set to 0 (null vector lift for conformal-like
      embeddings).

    After lifting, geodesic flow coherence is re-measured.  An improvement
    indicates that the extra dimension captures hidden geometric structure.
    """

    def __init__(self, device: str = "cpu", dtype: torch.dtype = CONSTANTS.default_dtype):
        self.device = device
        self.dtype = analysis_dtype(dtype)

    def lift(
        self,
        data: torch.Tensor,
        target_algebra: AlgebraLike,
        fill: float = CONSTANTS.dimension_lift_positive_fill,
    ) -> torch.Tensor:
        """Lifts data into the grade-1 subspace of a higher-dimensional algebra.

        Pads each data vector with ``fill`` values in the new dimensions,
        then embeds as a grade-1 multivector.

        Args:
            data: ``[N, d]`` source data.
            target_algebra: Target algebra with n >= d.
            fill: Coordinate value for the new dimensions.
                Use 1.0 for a projective (homogeneous) lift,
                0.0 for a null-vector (conformal) lift.

        Returns:
            ``[N, target_algebra.dim]`` grade-1 multivectors.
        """
        N, d = data.shape
        n = target_algebra.n
        dtype = getattr(target_algebra, "dtype", self.dtype)
        if n < d:
            raise ValueError(f"Target algebra dim {n} < source data dim {d}.")
        if n == d:
            return target_algebra.embed_vector(as_analysis_tensor(data, device=self.device, dtype=dtype))

        pad = torch.full(
            (N, n - d),
            fill,
            device=self.device,
            dtype=dtype,
        )
        lifted = torch.cat([as_analysis_tensor(data, device=self.device, dtype=dtype), pad], dim=-1)
        return target_algebra.embed_vector(lifted)

    def test(
        self,
        data: torch.Tensor,
        p: int,
        q: int,
        k: int = CONSTANTS.default_k_neighbors,
    ) -> Dict:
        """Compares geodesic flow coherence before and after dimension lifting.

        Tests three algebras:

        1. **Original** Cl(p, q): baseline coherence and curvature.
        2. **Positive lift** Cl(p+1, q): spacelike extra dimension, fill=1.
        3. **Negative-square lift** ``Cl(p, q+1)``: one additional negative
           generator, with its input coordinate initialized to zero. The
           returned key ``lift_null`` is retained for API compatibility; the
           added generator is not signature-null.

        Args:
            data: ``[N, d]`` data where d = p + q.
            p: Original positive signature.
            q: Original negative signature.
            k: Nearest neighbors for geodesic flow.

        Returns:
            Dict with keys ``original``, ``lift_positive``, ``lift_null``,
            and ``best`` (name of the algebra with highest coherence). The
            key ``lift_null`` denotes the negative-square lift described above.
        """
        from .geodesic import GeodesicFlow

        data = as_analysis_tensor(data, device=self.device, dtype=self.dtype)
        results: Dict = {}

        def _measure(alg: AlgebraLike, mv: torch.Tensor) -> Dict:
            gf = GeodesicFlow(alg, k=k)
            coh = gf.coherence(mv)
            curv = gf.curvature(mv)
            baseline = gf._random_baseline_coherence()
            coh_threshold = (1.0 + baseline) / 2.0
            return {
                "signature": (alg.p, alg.q),
                "coherence": coh,
                "curvature": curv,
                "causal": (coh > coh_threshold) and (curv < CONSTANTS.curvature_causal_threshold),
            }

        alg_orig = make_algebra(p, q, device=self.device, dtype=self.dtype)
        mv_orig = alg_orig.embed_vector(data[..., : alg_orig.n])
        results["original"] = _measure(alg_orig, mv_orig)

        alg_pos = make_algebra(p + 1, q, device=self.device, dtype=self.dtype)
        mv_pos = self.lift(data, alg_pos, fill=CONSTANTS.dimension_lift_positive_fill)
        results["lift_positive"] = _measure(alg_pos, mv_pos)

        alg_null = make_algebra(p, q + 1, device=self.device, dtype=self.dtype)
        mv_null = self.lift(data, alg_null, fill=CONSTANTS.dimension_lift_null_fill)
        results["lift_null"] = _measure(alg_null, mv_null)

        best = max(
            ("original", "lift_positive", "lift_null"),
            key=lambda key: results[key]["coherence"],
        )
        results["best"] = best

        return results

    def format_report(self, results: Dict) -> str:
        """Render a lifting test result.

        The legacy ``Causal`` and ``Noisy`` labels report only the coherence
        and curvature threshold computed by :meth:`test`; they do not establish
        causality or a noise model.
        """
        lines = ["Dimension Lifting Report", "=" * 40]
        for key in ("original", "lift_positive", "lift_null"):
            r = results[key]
            p, q = r["signature"]
            coh = r["coherence"]
            curv = r["curvature"]
            causal = "O Causal" if r["causal"] else "X Noisy"
            lines.append(f"  Cl({p},{q})  coherence={coh:+.3f}  curvature={curv:.3f}  {causal}")
        lines.append(f"\n  Best algebra: {results['best']}")
        return "\n".join(lines)
