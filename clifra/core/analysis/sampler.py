# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Statistical sampling strategies for geometric data analysis.

The stratified sampler uses geodesic-flow coherence as its stratification
score. It constructs a capped Euclidean algebra internally rather than
inferring a metric signature from the data.
"""

from typing import Dict, List, Tuple, Union

import torch

from ._types import CONSTANTS, SamplingConfig


class StatisticalSampler:
    """Stateless sampler supporting multiple strategies.

    Supports ``"random"``, ``"stratified"``, ``"bootstrap"``, and
    ``"passthrough"`` strategies.  All methods are deterministic when a
    seed is provided.

    The ``"stratified"`` strategy partitions per-point geodesic-flow
    coherence scores into quantile strata. It uses a capped Euclidean algebra
    for this calculation.
    """

    @staticmethod
    def sample(
        data: torch.Tensor,
        config: SamplingConfig,
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict]:
        """Sample from *data* according to *config*.

        Args:
            data: ``[N, D]`` tensor of observations.
            config: Sampling configuration.

        Returns:
            ``(sampled, metadata)`` where *sampled* is a tensor (or list
            of tensors for ``"bootstrap"``) and *metadata* is a dict with
            at least ``"strategy"``.
        """
        strategy = config.strategy
        if strategy == "passthrough":
            return data, {"strategy": "passthrough", "n_original": data.shape[0]}

        if strategy == "random":
            return StatisticalSampler._random(data, config)
        if strategy == "stratified":
            return StatisticalSampler._stratified(data, config)
        if strategy == "bootstrap":
            return StatisticalSampler._bootstrap(data, config)

        raise ValueError(f"Unknown sampling strategy: {strategy!r}")

    @staticmethod
    def recommend_size(n_features: int, n_total: int) -> int:
        """Heuristic recommendation for adequate sample size.

        Returns ``min(n_total, max(CONSTANTS.sampling_recommended_min_samples,
        CONSTANTS.sampling_recommended_feature_multiplier * n_features))``.
        """
        return min(
            n_total,
            max(
                CONSTANTS.sampling_recommended_min_samples,
                CONSTANTS.sampling_recommended_feature_multiplier * n_features,
            ),
        )

    @staticmethod
    def _random(data: torch.Tensor, config: SamplingConfig) -> Tuple[torch.Tensor, Dict]:
        N = data.shape[0]
        k = min(N, config.max_samples)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)
        indices = torch.randperm(N, generator=gen)[:k].to(data.device)
        return data[indices], {
            "strategy": "random",
            "indices": indices,
            "n_original": N,
        }

    @staticmethod
    def _stratified(data: torch.Tensor, config: SamplingConfig) -> Tuple[torch.Tensor, Dict]:
        """Coherence-based stratified sampling.

        Embeds data as grade-1 in a default Euclidean algebra, computes
        per-point geodesic-flow coherence, then partitions into quantile
        strata, then samples across those score ranges.
        """
        from clifra.core.config import make_algebra

        from .geodesic import GeodesicFlow

        N, D = data.shape
        k = min(N, config.max_samples)
        n_strata = config.n_strata or max(
            CONSTANTS.stratified_min_strata,
            min(k // CONSTANTS.stratified_points_per_stratum, CONSTANTS.stratified_max_strata),
        )
        gen = torch.Generator(device="cpu").manual_seed(config.seed)

        # Cap algebra dimension for tractability
        alg_dim = min(D, CONSTANTS.stratified_algebra_max_dim)
        dtype = data.dtype if data.dtype.is_floating_point else CONSTANTS.default_dtype
        algebra = make_algebra(alg_dim, 0, device=data.device, dtype=dtype)

        # Embed into algebra (truncate to alg_dim if needed)
        raw = data[:, :alg_dim].to(dtype=dtype)
        mv = algebra.embed_vector(raw)  # [N, 2^alg_dim]

        # Compute per-point coherence as the stratification metric
        gf_k = min(CONSTANTS.default_k_neighbors, N - 1)
        gf = GeodesicFlow(algebra, k=gf_k)
        with torch.no_grad():
            scores = gf.per_point_coherence(mv)  # [N]

        # Assign to quantile strata
        quantiles = torch.linspace(0, 1, n_strata + 1, device=data.device, dtype=scores.dtype)
        thresholds = torch.quantile(scores, quantiles)

        labels = torch.bucketize(scores, thresholds[1:-1]).clamp(max=n_strata - 1)

        # Proportional draw from each stratum
        all_indices: List[torch.Tensor] = []
        for c in range(n_strata):
            mask = (labels == c).nonzero(as_tuple=True)[0]
            if len(mask) == 0:
                continue
            n_from = max(1, int(round(k * len(mask) / N)))
            n_from = min(n_from, len(mask))
            perm = torch.randperm(len(mask), generator=gen)[:n_from].to(mask.device)
            all_indices.append(mask[perm])

        indices = torch.cat(all_indices) if all_indices else torch.empty(0, dtype=torch.long, device=data.device)
        if len(indices) > k:
            indices = indices[:k]

        return data[indices], {
            "strategy": "stratified",
            "indices": indices,
            "n_strata": n_strata,
            "n_original": N,
            "coherence_scores": scores,
        }

    @staticmethod
    def _bootstrap(data: torch.Tensor, config: SamplingConfig) -> Tuple[List[torch.Tensor], Dict]:
        N = data.shape[0]
        k = min(N, config.max_samples)
        gen = torch.Generator(device="cpu").manual_seed(config.seed)

        resamples: List[torch.Tensor] = []
        for _ in range(config.n_bootstrap):
            indices = torch.randint(0, N, (k,), generator=gen).to(data.device)
            resamples.append(data[indices])

        return resamples, {
            "strategy": "bootstrap",
            "n_bootstrap": config.n_bootstrap,
            "sample_size": k,
            "n_original": N,
        }
