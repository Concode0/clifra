# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


import torch

from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import AlgebraLike, CliffordModule
from clifra.core.foundation.numerics import signed_clamp_min
from clifra.core.storage import resolve_layer_layout

from ._layout import basis_positions


class ConformalEmbedding(CliffordModule):
    """Example conformal embedding over a declared active-lane layout.

    Maps Euclidean R^d vectors to grade-1 conformal points in Cl(d+1, 1) and
    back. The output lane width is ``layout.dim`` rather than always
    ``algebra.dim``.

    The standard conformal embedding is:
        P(x) = x + 0.5 * ||x||^2 * e_inf + e_o

    where e_inf = e_- + e_+ (point at infinity) and e_o = 0.5*(e_- - e_+)
    (origin). Basis vectors e_inf and e_o are precomputed as buffers.

    Requires an algebra with signature Cl(d+1, 1, ...).

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
        """Sets up the conformal embedding.

        Args:
            algebra: Clifford algebra instance with signature Cl(d+1, 1, ...).
            euclidean_dim: Physical dimension d.
            grades: Optional compact output/input grades.
            layout: Optional compact output/input layout.
        """
        super().__init__(algebra)
        d = euclidean_dim
        if algebra.p < d + 1 or algebra.q < 1:
            raise ValueError(
                f"Conformal embedding needs Cl(>={d + 1}, >=1), got Cl({algebra.p},{algebra.q},{algebra.r})"
            )
        self.euclidean_dim = d
        self.layout = resolve_layer_layout(algebra, layout=layout, grades=grades)
        self.scalar_layout = algebra.layout((0,))
        self.lane_dim = self.layout.dim

        idx_ep = 1 << d
        idx_em = 1 << (d + 1)
        g1_dense = [1 << bit for bit in range(d)]
        self.register_buffer("_g1_idx", basis_positions(self.layout, g1_dense, name="ConformalEmbedding"))

        ep_pos, em_pos = basis_positions(self.layout, (idx_ep, idx_em), name="ConformalEmbedding").tolist()
        e_inf = torch.zeros(self.lane_dim, dtype=algebra.dtype)
        e_inf[em_pos] = 1.0
        e_inf[ep_pos] = 1.0
        self.register_buffer("_e_inf", e_inf)

        e_o = torch.zeros(self.lane_dim, dtype=algebra.dtype)
        e_o[em_pos] = 0.5
        e_o[ep_pos] = -0.5
        self.register_buffer("_e_o", e_o)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed Euclidean points into the conformal null cone.

        Args:
            x: Euclidean points [..., d].

        Returns:
            Conformal multivectors [..., layout.dim].
        """
        d = self.euclidean_dim
        if x.shape[-1] != d:
            raise ValueError(f"input last dimension must be {d}, got {x.shape[-1]}")
        x_mv = torch.zeros(*x.shape[:-1], self.lane_dim, device=x.device, dtype=x.dtype)
        g1_idx = self._g1_idx.to(device=x.device)
        x_mv.scatter_(-1, g1_idx.expand_as(x), x)

        x_sq = (x * x).sum(dim=-1, keepdim=True)  # [..., 1]
        e_inf = self._e_inf.to(device=x.device, dtype=x.dtype)
        e_o = self._e_o.to(device=x.device, dtype=x.dtype)
        return x_mv + 0.5 * x_sq * e_inf + e_o

    def extract(self, P: torch.Tensor) -> torch.Tensor:
        """Project conformal points back to Euclidean space.

        Normalizes so that P . e_inf = -1, then extracts vector part.

        Args:
            P: Conformal multivectors [..., layout.dim].

        Returns:
            Euclidean coordinates [..., d].
        """
        d = self.euclidean_dim
        if P.shape[-1] != self.lane_dim:
            raise ValueError(f"point last dimension must be {self.lane_dim}, got {P.shape[-1]}")

        e_inf = self._e_inf.to(device=P.device, dtype=P.dtype)
        P_einf = self.algebra.geometric_product(
            P,
            e_inf.expand_as(P),
            left_layout=self.layout,
            right_layout=self.layout,
            output_layout=self.scalar_layout,
        )
        scale = signed_clamp_min(-P_einf[..., 0:1], self.algebra.eps)
        P_norm = P / scale

        return torch.gather(P_norm, -1, self._g1_idx.to(P.device).expand(*P.shape[:-1], d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default forward: embed Euclidean points.

        Args:
            x: Euclidean points [..., d].

        Returns:
            Conformal multivectors [..., layout.dim].
        """
        return self.embed(x)
