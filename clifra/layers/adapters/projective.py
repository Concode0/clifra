# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import signed_clamp_min
from clifra.core.storage import resolve_layer_layout

from ._layout import basis_positions


class ProjectiveEmbedding(CliffordModule):
    """Example projective embedding over a declared active-lane layout.

    Maps Euclidean R^d points to grade-1 elements of Cl(d, 0, 1) and back.
    The output lane width is ``layout.dim`` rather than always ``algebra.dim``.

    In PGA, the degenerate basis vector e_0 (where e_0^2 = 0) represents
    the origin. A Euclidean point x is embedded as the 1-vector:

        P(x) = x_1 e_1 + x_2 e_2 + ... + x_d e_d + e_0

    The e_0 component acts as a homogeneous coordinate. Directions (ideal
    points) have e_0 = 0, while finite points have e_0 != 0.

    Extraction normalizes so that the e_0 coefficient equals 1, then
    reads the Euclidean basis coefficients.

    Requires an algebra with signature Cl(d, 0, >=1).

    Attributes:
        euclidean_dim (int): Physical dimension d.
    """

    def __init__(
        self,
        algebra: AlgebraLike,
        euclidean_dim: int,
        *,
        grades=None,
        layout: GradeLayout = None,
    ):
        """Sets up the projective embedding.

        Args:
            algebra: Clifford algebra instance with signature Cl(d, 0, >=1).
            euclidean_dim: Physical dimension d.
            grades: Optional compact output/input grades.
            layout: Optional compact output/input layout.
        """
        super().__init__(algebra)
        d = euclidean_dim
        if algebra.p < d or algebra.r < 1:
            raise ValueError(
                f"Projective embedding needs Cl(>={d}, 0, >=1), got Cl({algebra.p},{algebra.q},{algebra.r})"
            )
        self.euclidean_dim = d
        self.layout = resolve_layer_layout(algebra, layout=layout, grades=grades)
        self.lane_dim = self.layout.dim

        g1_dense = [1 << bit for bit in range(d)]
        self.register_buffer("_g1_idx", basis_positions(self.layout, g1_dense, name="ProjectiveEmbedding"))

        idx_e0 = 1 << (algebra.p + algebra.q)
        e0_pos = basis_positions(self.layout, (idx_e0,), name="ProjectiveEmbedding")[0]
        self.register_buffer("_idx_e0", e0_pos.clone().detach())

        e0 = torch.zeros(self.lane_dim, dtype=algebra.dtype)
        e0[int(e0_pos)] = 1.0
        self.register_buffer("_e0", e0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean points into PGA as grade-1 elements.

        Args:
            x: Euclidean points [..., d].

        Returns:
            PGA multivectors [..., layout.dim].
        """
        if x.shape[-1] != self.euclidean_dim:
            raise ValueError(f"input last dimension must be {self.euclidean_dim}, got {x.shape[-1]}")
        x_mv = torch.zeros(*x.shape[:-1], self.lane_dim, device=x.device, dtype=x.dtype)
        x_mv.scatter_(-1, self._g1_idx.to(x.device).expand_as(x), x)
        return x_mv + self._e0.to(device=x.device, dtype=x.dtype)

    def embed_direction(self, v: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean directions (ideal points) into PGA.

        Directions have e_0 = 0, representing points at infinity.

        Args:
            v: Direction vectors [..., d].

        Returns:
            PGA multivectors [..., layout.dim] with zero e_0 component.
        """
        if v.shape[-1] != self.euclidean_dim:
            raise ValueError(f"direction last dimension must be {self.euclidean_dim}, got {v.shape[-1]}")
        x_mv = torch.zeros(*v.shape[:-1], self.lane_dim, device=v.device, dtype=v.dtype)
        x_mv.scatter_(-1, self._g1_idx.to(v.device).expand_as(v), v)
        return x_mv

    def extract(self, P: torch.Tensor) -> torch.Tensor:
        """Project PGA points back to Euclidean space.

        Normalizes by the e_0 coefficient, then extracts the
        Euclidean basis components.

        Args:
            P: PGA multivectors [..., layout.dim].

        Returns:
            Euclidean coordinates [..., d].
        """
        d = self.euclidean_dim
        if P.shape[-1] != self.lane_dim:
            raise ValueError(f"point last dimension must be {self.lane_dim}, got {P.shape[-1]}")
        # Normalize by e_0 coefficient (homogeneous coordinate)
        e0_pos = int(self._idx_e0.item())
        e0_coeff = signed_clamp_min(P[..., e0_pos : e0_pos + 1], self.algebra.eps)
        P_norm = P / e0_coeff
        return torch.gather(P_norm, -1, self._g1_idx.to(P.device).expand(*P.shape[:-1], d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: embed Euclidean points.

        Args:
            x: Euclidean points [..., d].

        Returns:
            PGA multivectors [..., layout.dim].
        """
        return self.embed(x)
