# Versor: Universal Geometric Algebra Neural Network
# Copyright (C) 2026 Eunkyum Kim <nemonanconcode@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Geodesic flow analysis in Clifford algebra.

Interprets data points as grade-1 multivectors and computes the *flow
field* -- a bivector at each point encoding the direction of shortest
algebraic paths to its k-nearest neighbours.
"""

from typing import Dict

import torch

from clifra.core.foundation.module import AlgebraLike
from clifra.core.foundation.numerics import eps_like

from ._types import CONSTANTS
from ._utils import require_dense_algebra


class GeodesicFlow:
    """Geodesic flow analysis in Clifford algebra.

    Interprets data points as grade-1 multivectors and computes the *flow
    field* -- a bivector at each point that encodes the direction of shortest
    algebraic paths to its k-nearest neighbours.

    The flow is computed as the mean of *connection bivectors*:

        B_ij = <x_i . ~x_j>_2   (grade-2 part of the geometric product)

    This bivector encodes the rotational "turn" needed to map x_i toward
    x_j, analogous to the parallel transport connection on a Lie group.

    The coherence and curvature of this field reveal whether the data has
    causal (directional) structure:

    - **High coherence, low curvature** -> the flow is smooth and aligned in
      one direction.  Causality is visible.
    - **Low coherence, high curvature** -> the flow is fragmented and
      collides with itself.  The signal is dominated by noise.
    """

    def __init__(self, algebra: AlgebraLike, k: int = CONSTANTS.default_k_neighbors):
        """Initialize Geodesic Flow.

        Args:
            algebra (CliffordAlgebra): The algebra instance.
            k (int): Number of nearest neighbours for the flow field.
        """
        self.algebra = algebra
        self.k = int(k)

    def _embed(self, data: torch.Tensor) -> torch.Tensor:
        """Embeds raw vectors into the grade-1 multivector subspace.

        Args:
            data (torch.Tensor): ``[N, d]`` where d <= algebra.n.

        Returns:
            torch.Tensor: ``[N, algebra.dim]`` grade-1 multivectors.
        """
        N, d = data.shape
        n = self.algebra.n
        if d > n:
            raise ValueError(
                f"Data dimension {d} exceeds algebra dimension {n}. "
                f"Use DimensionLifter to lift data before flow analysis."
            )
        if d < n:
            pad = torch.zeros(N, n - d, device=data.device, dtype=data.dtype)
            data = torch.cat([data, pad], dim=-1)
        return self.algebra.embed_vector(data)

    def _knn(self, mv: torch.Tensor) -> torch.Tensor:
        """Returns k-nearest neighbour indices in multivector coefficient space.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: ``[N, k]`` neighbour indices.
        """
        N = mv.shape[0]
        k = min(self.k, N - 1)
        if k <= 0:
            return torch.empty(N, 0, dtype=torch.long, device=mv.device)
        dists = torch.cdist(mv, mv)  # [N, N]
        dists.fill_diagonal_(float("inf"))
        _, idx = dists.topk(k, dim=-1, largest=False)
        return idx  # [N, k]

    def _connection_bivectors(self, mv: torch.Tensor, *, active_output: bool = False) -> torch.Tensor:
        """Computes unit connection bivectors for all (point, neighbour) pairs.

        The connection bivector from x_i to x_j encodes the rotational "turn"
        in the algebra needed to map one vector toward the other:

            B_ij = unit( <x_i . ~x_j>_2 )

        Args:
            mv (torch.Tensor): ``[N, dim]`` grade-1 multivectors.

        Returns:
            torch.Tensor: ``[N, k, dim]`` unit connection bivectors.
        """
        N, D = mv.shape
        k = min(self.k, N - 1)
        if self.algebra.n < 2:
            width = 0 if active_output else D
            return mv.new_zeros(N, 0, width)
        layout = self.algebra.layout((2,))
        if k <= 0 or layout.dim == 0:
            width = layout.dim if active_output else D
            return mv.new_zeros(N, 0, width)
        nn_idx = self._knn(mv)

        neighbors = mv[nn_idx]  # [N, k, dim]
        xi = mv.unsqueeze(1).expand(N, k, D).reshape(N * k, D)
        xj_rev = neighbors.reshape(N * k, D)

        # For grade-1 inputs, <xi * ~xj>_2 = wedge(xi, xj_rev) -- single pass
        bv_raw = self.algebra.wedge(
            xi,
            xj_rev,
            left_grades=(1,),
            right_grades=(1,),
            output_grades=(2,),
            active_output=True,
        )  # [N*k, dim]
        bv_norm = bv_raw.norm(dim=-1, keepdim=True).clamp_min(eps_like(bv_raw))
        compact = (bv_raw / bv_norm).reshape(N, k, layout.dim)
        if active_output:
            return compact
        return layout.dense(compact)  # [N, k, dim]

    def flow_bivectors(self, mv: torch.Tensor) -> torch.Tensor:
        """Computes the mean flow bivector at each data point.

        For each point x_i, aggregates the unit connection bivectors to its
        k-nearest neighbours:

            B_i = mean_j { unit( <x_i . ~x_j>_2 ) }

        .. note::
            For perfectly symmetric data (e.g. a closed circle) the mean
            cancels to zero -- which is geometrically correct since there is
            no preferred flow direction.  Use :meth:`coherence` to measure
            structure without this cancellation.

        Args:
            mv (torch.Tensor): ``[N, dim]`` grade-1 multivectors.

        Returns:
            torch.Tensor: ``[N, dim]`` mean flow bivectors.
        """
        bv = self._connection_bivectors(mv)  # [N, k, dim]
        if bv.shape[1] == 0:
            return mv.new_zeros(mv.shape[0], mv.shape[-1])
        return bv.mean(dim=1)  # [N, dim]

    def _coherence_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        """Differentiable coherence -- returns a scalar tensor with grad_fn.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: Scalar coherence in [0, 1].
        """
        bv = self._connection_bivectors(mv, active_output=True)  # [N, k, grade2_dim]
        N, k, D = bv.shape
        if k < 2 or D == 0:
            return mv.new_zeros(())

        bi = bv.unsqueeze(2)  # [N, k, 1, dim]
        bj = bv.unsqueeze(1)  # [N, 1, k, dim]
        abs_cos = (bi * bj).sum(dim=-1).abs()  # [N, k, k]

        mask = ~torch.eye(k, dtype=torch.bool, device=mv.device)  # [k, k]
        off_diag = abs_cos[:, mask]  # [N, k*(k-1)]
        return off_diag.mean() if off_diag.numel() > 0 else mv.new_zeros(())

    def coherence(self, mv: torch.Tensor) -> float:
        """Measures concentration of connection bivectors within each neighbourhood.

        For each point, computes the mean **absolute** cosine similarity between
        all pairs of its k connection bivectors.  This captures how consistently
        the neighbourhood connections lie on the same rotation plane.

        - **1.0**: all connections at every point are parallel or anti-parallel
          (maximally structured).
        - **1/num_bivectors** (~= baseline): connections point in random directions.

        .. note::
            In Cl(2,0) the grade-2 space is 1-dimensional (only e_12), so
            coherence is trivially 1.0 for any data -- use at least Cl(3,0)
            for meaningful discrimination.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            float: Coherence score in [0, 1].
        """
        return self._coherence_tensor(mv).item()

    def _curvature_tensor(self, mv: torch.Tensor) -> torch.Tensor:
        """Differentiable curvature -- returns a scalar tensor with grad_fn.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: Scalar curvature in [0, 1].
        """
        bv = self._connection_bivectors(mv, active_output=True)  # [N, k, grade2_dim]
        N, k, D = bv.shape
        if k == 0 or D == 0:
            return mv.new_zeros(())
        nn_idx = self._knn(mv)  # [N, k_nn]

        bi = bv.unsqueeze(2)  # [N, k, 1, dim]
        bj_all = bv[nn_idx]  # [N, k_nn, k, dim]

        bj = bj_all[:, 0]  # [N, k, dim]
        bj = bj.unsqueeze(1)  # [N, 1, k, dim]

        cross_cos = (bi * bj).sum(dim=-1).abs()  # [N, k, k]
        alignment = cross_cos.mean(dim=(-1, -2))  # [N]

        return (1.0 - alignment.mean()).clamp_min(0.0)

    def curvature(self, mv: torch.Tensor) -> float:
        """Measures how much connection structure changes across the manifold.

        Computes the mean **dissimilarity** of connection bivectors between
        neighbouring pairs of points:

            dissimilarity(i, j) = 1 - mean_abs_cos( {B_ia}, {B_jb} )

        where {B_ia} is the set of k unit connection bivectors at point i and
        {B_jb} at point j, and mean_abs_cos is the cross-set absolute cosine
        similarity.

        - **0.0**: all neighbouring points have the same connection structure
          (flat geodesics, smooth manifold).
        - **High**: the connection direction changes rapidly between neighbours
          (high curvature, fragmented flow).

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            float: Curvature score in [0, 1].
        """
        return self._curvature_tensor(mv).item()

    def interpolate(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        steps: int = CONSTANTS.geodesic_interpolation_steps,
    ) -> torch.Tensor:
        """Interpolates along the geodesic from ``a`` to ``b``.

        Uses the *Lie group exponential map* on the transition element:

            T = a_inv . b
            log(T) ~= <T - 1>_2     (grade-2 approximation for small angles)
            gamma(t) = a . exp(t . log(T))

        Exact when a and b are close; a first-order approximation otherwise.

        Args:
            a (torch.Tensor): Start multivector ``[dim]``.
            b (torch.Tensor): End multivector ``[dim]``.
            steps (int): Number of interpolation steps (including endpoints).

        Returns:
            torch.Tensor: ``[steps, dim]`` sequence of multivectors.
        """
        require_dense_algebra(self.algebra, "GeodesicFlow.interpolate")
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")

        a = a.unsqueeze(0)  # [1, dim]
        b = b.unsqueeze(0)  # [1, dim]

        a_inv = self.algebra.blade_inverse(a)  # [1, dim]

        # Transition element T = a_inv . b
        T = self.algebra.geometric_product(a_inv, b)  # [1, dim]

        # Log approximation: grade-2 part of (T - 1)
        T_shift = T.clone()
        T_shift[..., 0] -= 1.0
        log_T = self.algebra.grade_projection(T_shift, 2)  # [1, dim]

        # Sample t in [0, 1]
        ts = torch.linspace(0.0, 1.0, steps, device=a.device, dtype=a.dtype)

        # Batched: scale log_T by all t values, exp, then GP
        scaled = ts.unsqueeze(-1) * log_T  # [steps, dim]
        exp_all = self.algebra.exp(scaled)  # [steps, dim]
        return self.algebra.geometric_product(a, exp_all)  # [steps, dim]

    def _random_baseline_coherence(self) -> float:
        """Expected coherence for random unit vectors in the bivector subspace.

        For d-dimensional random unit vectors, ``E[|cos theta|] ~= sqrt(2/(pi*d))``.
        In Cl(n), the bivector space has ``n*(n-1)/2`` dimensions.
        """
        import math

        n = self.algebra.n
        d = n * (n - 1) // 2
        if d <= 1:
            return 1.0
        return math.sqrt(2.0 / (math.pi * d))

    def causal_report(self, data: torch.Tensor) -> Dict:
        """Full geodesic flow analysis with a causal interpretation.

        Embeds data, computes coherence and curvature, and returns a
        human-readable verdict.

        The causal threshold is adaptive: coherence must exceed the
        midpoint between random baseline and 1.0 (i.e. the measured
        coherence must be at least halfway between chance and perfect
        alignment).  Curvature must be below 0.5.

        Args:
            data (torch.Tensor): ``[N, d]`` raw data.

        Returns:
            Dict: report with keys ``coherence``, ``curvature``,
            ``causal``, ``baseline``, ``threshold``, ``label``.
        """
        mv = self._embed(data)
        coh = self.coherence(mv)
        curv = self.curvature(mv)
        baseline = self._random_baseline_coherence()
        threshold = (1.0 + baseline) / 2.0
        is_causal = (coh > threshold) and (curv < CONSTANTS.curvature_causal_threshold)
        return {
            "coherence": coh,
            "curvature": curv,
            "baseline": baseline,
            "threshold": threshold,
            "causal": is_causal,
            "label": (
                "Causal - smooth, aligned flow (low curvature)"
                if is_causal
                else "Noisy - fragmented, colliding flow (high curvature)"
            ),
        }

    def per_point_coherence(self, mv: torch.Tensor) -> torch.Tensor:
        """Per-point coherence scores for stratified sampling.

        Returns a scalar coherence value per data point, measuring how
        well-aligned that point's neighbourhood connections are.

        Args:
            mv (torch.Tensor): ``[N, dim]`` multivectors.

        Returns:
            torch.Tensor: ``[N]`` coherence scores in [0, 1].
        """
        bv = self._connection_bivectors(mv, active_output=True)  # [N, k, grade2_dim]
        N, k, D = bv.shape
        if k < 2 or D == 0:
            return mv.new_zeros(N)

        bi = bv.unsqueeze(2)  # [N, k, 1, dim]
        bj = bv.unsqueeze(1)  # [N, 1, k, dim]
        abs_cos = (bi * bj).sum(dim=-1).abs()  # [N, k, k]

        mask = ~torch.eye(k, dtype=torch.bool, device=mv.device)  # [k, k]
        # Mean over off-diagonal pairs per point
        off_diag = abs_cos[:, mask].reshape(N, -1)  # [N, k*(k-1)]
        return off_diag.mean(dim=1) if off_diag.shape[-1] > 0 else mv.new_zeros(N)  # [N]
