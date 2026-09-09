# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Active signature evidence from a fixed quadratic-lift probe workflow.

Probes train in Cl(D+1,1); the selected probe's bivector parameter energies are mapped to a
heuristic active (p, q) pair. Unresolved features are counted separately and
are not interpreted as null directions.
"""

import concurrent.futures
import copy
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn as nn

from clifra.core._kernel.basis import build_bivector_squared_signs
from clifra.core._kernel.composition import full_versor_factors
from clifra.core._kernel.device import resolve_dtype
from clifra.core._kernel.execution.action import FullSandwichActionExecutor
from clifra.core._kernel.planning.resources import ResourceLimits
from clifra.core.algebra import AlgebraContext
from clifra.core.config import make_algebra
from clifra.core.layout import AlgebraSpec
from clifra.core.module import CliffordModule

from .._resources import check_matrix_budget
from .neighborhood import NeighborhoodBivectorAnalyzer

__all__ = ["SignatureProbeAnalyzer", "SignatureProbeResult"]

_PROBE_LIMITS = ResourceLimits(max_lanes=1 << 12, max_pairs=1 << 24)
_PROBE_CHANNELS = 4
_DISSIMILARITY_WEIGHT = 0.3
_L1_WEIGHT = 0.01
_BV_SQ_ELLIPTIC_BOUND = -0.5
_BV_SQ_HYPERBOLIC_BOUND = 0.5
_SIGNATURE_PROBE_BIAS_MINOR_WEIGHT = 0.1
_SIGNATURE_PROBE_BIAS_NOISE_STD = 0.05
_SIGNATURE_PROBE_PARALLEL_WORKER_CAP = 4
_UNIFORM_INIT_BOUND = 0.5
_NORMAL_INIT_STD = 0.3
_SIGNATURE_PROBE_ROTOR_INIT_STD = 0.01


@dataclass
class SignatureProbeResult:
    """Heuristic active (p, q) evidence, unresolved features, and probe scores.

    ``unresolved_features`` is the probe feature width minus p+q, after any PCA.
    It includes inactive/unassigned features, not null directions. Discarded PCA
    coordinates are outside this count. ``pca_output_width`` is the actual PCA
    width, or None without reduction.
    ``bivector_parameter_summary`` contains channel-mean squared bivector parameters,
    normalized by their maximum, and counts of active coordinates dominated by
    elliptic or hyperbolic bivector contributions. These are weight
    classifications, not generator signs. Per-probe ``best_training_loss`` is
    the lowest recorded training objective; neighborhood scores are evaluated
    after restoring the selected parameters. Each probe record includes its
    ``init_mode``; list order identifies repeated modes.
    """

    candidate_active_pq: tuple[int, int]
    neighbor_bivector_alignment: float
    neighbor_bivector_dissimilarity: float
    bivector_parameter_summary: dict
    unresolved_features: int
    pca_output_width: int | None = None
    per_probe_results: list[dict] = field(default_factory=list)


def _pca_reduce(data: torch.Tensor, width: int) -> torch.Tensor:
    centered = data - data.mean(dim=0, keepdim=True)
    u, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    return u[:, :width] * singular[:width]


class _ProbeChannelMixer(CliffordModule):
    """Affine channel mixer with a separate bias for each output coefficient."""

    def __init__(self, algebra: AlgebraContext, in_channels: int, out_channels: int):
        super().__init__(algebra)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels, algebra.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("oi,...id->...od", self.weight, x) + self.bias


class _ProbeRotor(CliffordModule):
    """Probe-local full-lane rotor for signature probing."""

    def __init__(self, algebra: AlgebraContext, channels: int):
        super().__init__(algebra)
        self.channels = channels
        self.parameter_layout = algebra.layout((2,))
        self.full_layout = algebra.layout()
        self.action = FullSandwichActionExecutor.from_layout(
            self.full_layout,
            device=algebra.device,
            dtype=algebra.dtype,
        )
        self.bivector_parameters = nn.Parameter(torch.empty(channels, self.parameter_layout.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bivector_parameters, std=_SIGNATURE_PROBE_ROTOR_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = full_versor_factors(
            self.algebra,
            self.bivector_parameters.to(device=x.device, dtype=x.dtype),
            grade=2,
            parameter_layout=self.parameter_layout,
        )
        return self.action(left, x, right)

    def parameter_l1_penalty(self) -> torch.Tensor:
        return torch.norm(self.bivector_parameters, p=1)


class _ProbeBladeGate(CliffordModule):
    """Continuous sigmoid gate for each channel and blade coefficient."""

    def __init__(self, algebra: AlgebraContext, channels: int):
        super().__init__(algebra)
        self.weights = nn.Parameter(torch.ones(channels, algebra.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.weights).unsqueeze(0)


class _SignatureProbe(nn.Module):
    """Single-rotor probe whose bivector parameter energies map to a (p, q) candidate.

    Architecture: channel mixer -> rotor -> channel mixer -> blade gate.
    """

    def __init__(self, algebra: AlgebraContext, channels: int = _PROBE_CHANNELS):
        super().__init__()
        self.algebra = algebra
        self.channel_expansion = _ProbeChannelMixer(algebra, 1, channels)
        self.rotor = _ProbeRotor(algebra, channels)
        self.channel_reduction = _ProbeChannelMixer(algebra, channels, 1)
        self.gate = _ProbeBladeGate(algebra, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_expansion(x)
        x = self.rotor(x)
        x = self.channel_reduction(x)
        x = self.gate(x)
        return x


def _initialize_bivector_parameters(
    probe: _SignatureProbe,
    algebra: AlgebraContext,
    init_mode: str = "normal",
) -> None:
    """Initialize bivector parameters using declared basis-bivector square signs.

    Uses static basis signature data to classify each basis bivector:
    - bv_sq = -1: elliptic (equal-sign non-null base vectors)
    - bv_sq = +1: hyperbolic (mixed-signature base vectors)
    - bv_sq =  0: null (degenerate base vectors)
    """
    bv_sq = build_bivector_squared_signs(algebra.layout((2,)), device=algebra.device, dtype=algebra.dtype)
    ell = _BV_SQ_ELLIPTIC_BOUND
    hyp = _BV_SQ_HYPERBOLIC_BOUND
    rotor = probe.rotor
    with torch.no_grad():
        if init_mode == "elliptic_weighted":
            weights = torch.where(
                bv_sq < ell,
                torch.ones_like(bv_sq),
                torch.full_like(bv_sq, _SIGNATURE_PROBE_BIAS_MINOR_WEIGHT),
            )
            rotor.bivector_parameters.copy_(
                weights.unsqueeze(0).expand_as(rotor.bivector_parameters)
                + torch.randn_like(rotor.bivector_parameters) * _SIGNATURE_PROBE_BIAS_NOISE_STD
            )
        elif init_mode == "non_null_weighted":
            weights = torch.where(
                bv_sq.abs() > hyp,
                torch.ones_like(bv_sq),
                torch.full_like(bv_sq, _SIGNATURE_PROBE_BIAS_MINOR_WEIGHT),
            )
            rotor.bivector_parameters.copy_(
                weights.unsqueeze(0).expand_as(rotor.bivector_parameters)
                + torch.randn_like(rotor.bivector_parameters) * _SIGNATURE_PROBE_BIAS_NOISE_STD
            )
        elif init_mode == "uniform":
            nn.init.uniform_(
                rotor.bivector_parameters,
                -_UNIFORM_INIT_BOUND,
                _UNIFORM_INIT_BOUND,
            )
        else:  # 'normal'
            nn.init.normal_(rotor.bivector_parameters, 0.0, _NORMAL_INIT_STD)


class SignatureProbeAnalyzer:
    """Probe signature candidates, with optional capped PCA and bootstrap voting.

    ``max_probe_features`` caps the feature count through PCA before lifting.
    Probe counts/epochs/rate control training work; k controls neighborhoods.
    ``bivector_parameter_energy_threshold`` thresholds normalized bivector parameter energies.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_probe_features: int = 10,
        num_probes: int = 6,
        probe_epochs: int = 80,
        probe_lr: float = 0.005,
        k: int = 8,
        bivector_parameter_energy_threshold: float = 0.05,
    ):
        for name, value in (
            ("max_probe_features", max_probe_features),
            ("num_probes", num_probes),
            ("probe_epochs", probe_epochs),
            ("k", k),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < probe_lr < float("inf") or not 0 <= bivector_parameter_energy_threshold <= 1:
            raise ValueError(
                "probe_lr must be positive and finite; bivector_parameter_energy_threshold must be in [0,1]"
            )
        self.device = device
        self.dtype = resolve_dtype(dtype)
        if not self.dtype.is_floating_point:
            raise TypeError("signature probing requires a floating-point dtype")
        self.max_probe_features = max_probe_features
        self.num_probes = num_probes
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.k = k
        self.bivector_parameter_energy_threshold = bivector_parameter_energy_threshold

    def analyze(self, data: torch.Tensor) -> SignatureProbeResult:
        """Probe nonempty raw [N,D] observations, detached from the caller's graph."""
        if data.ndim != 2 or min(data.shape) < 1:
            raise ValueError("data must have nonempty shape [N,D]")
        data = data.detach().to(device=self.device, dtype=self.dtype)
        reduced_width = None
        if data.shape[1] > self.max_probe_features:
            data = _pca_reduce(data, self.max_probe_features)
            reduced_width = data.shape[1]
        result = self._run_probes(data)
        return SignatureProbeResult(**result, pca_output_width=reduced_width)

    def analyze_bootstrap(
        self, data: torch.Tensor, *, n_bootstrap: int = 10, max_samples: int = 500, seed: int = 42
    ) -> tuple[SignatureProbeResult, dict]:
        """Return a modal representative and bootstrap signature counts.

        The returned result is the first replicate with the modal tuple,
        not a refit on the original data. The seed controls resampling indices;
        probe training still uses PyTorch's RNG. Ties favor the first seen tuple.
        """
        if data.ndim != 2 or min(data.shape) < 1:
            raise ValueError("data must have nonempty shape [N,D]")
        for name, value in (("n_bootstrap", n_bootstrap), ("max_samples", max_samples)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        results = []
        for _ in range(n_bootstrap):
            indices = torch.randint(len(data), (min(len(data), max_samples),), generator=generator).to(data.device)
            results.append(self.analyze(data[indices]))
        votes = Counter(result.candidate_active_pq for result in results)
        signature, count = votes.most_common(1)[0]
        representative = next(result for result in results if result.candidate_active_pq == signature)
        return representative, {
            "candidate_active_pq_counts": dict(votes),
            "modal_fraction": count / n_bootstrap,
            "n_bootstrap": n_bootstrap,
        }

    def _quadratic_lift(self, data: torch.Tensor) -> Tuple[torch.Tensor, AlgebraContext]:
        """Lifts [N, X] data into Cl(X+1, 1, 0) via a quadratic coordinate embedding."""
        data = data.to(device=self.device, dtype=self.dtype)
        N, X = data.shape

        spec = AlgebraSpec(X + 1, 1, 0)
        action_budget = check_matrix_budget(
            role="signature_probe",
            matrix_dim=spec.dim,
            limits=_PROBE_LIMITS,
            matrix_kind="action",
            dtype=data.dtype,
        )
        if not action_budget:
            raise ValueError(f"signature probe action matrix exceeds resource limits: {dict(action_budget.details)}")

        half_radius_squared = 0.5 * (data**2).sum(dim=-1, keepdim=True)
        ones = torch.ones(N, 1, device=self.device, dtype=data.dtype)
        lifted = torch.cat([data, half_radius_squared, ones], dim=-1)

        algebra = make_algebra(X + 1, 1, 0, device=self.device, dtype=data.dtype)
        mv = algebra.layout((1,)).full(lifted)
        mv = mv.unsqueeze(1)
        return mv, algebra

    def _train_probe(
        self,
        mv_data: torch.Tensor,
        algebra: AlgebraContext,
        init_mode: str = "normal",
    ) -> Dict:
        """Trains a single probe and returns results."""
        probe = _SignatureProbe(algebra, channels=_PROBE_CHANNELS)
        probe.to(device=self.device, dtype=self.dtype)
        _initialize_bivector_parameters(probe, algebra, init_mode)

        neighborhood = NeighborhoodBivectorAnalyzer(algebra, k=self.k)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        best_loss = float("inf")
        best_state = None

        for _ in range(self.probe_epochs):
            optimizer.zero_grad()
            output = probe(mv_data)
            output_flat = output.squeeze(1)

            bivector_alignment_t = neighborhood._bivector_alignment_tensor(output_flat)
            bivector_dissimilarity_t = neighborhood._bivector_dissimilarity_tensor(output_flat)

            l1_penalty = probe.rotor.parameter_l1_penalty()

            loss = -bivector_alignment_t + _DISSIMILARITY_WEIGHT * bivector_dissimilarity_t + _L1_WEIGHT * l1_penalty

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = copy.deepcopy(probe.state_dict())

            loss.backward()
            optimizer.step()

        if best_state is not None:
            probe.load_state_dict(best_state)

        with torch.no_grad():
            output = probe(mv_data).squeeze(1)
            alignment = neighborhood.neighbor_bivector_alignment(output)
            dissimilarity = neighborhood.neighbor_bivector_dissimilarity(output)

        return {
            "best_training_loss": best_loss,
            "neighbor_bivector_alignment": alignment,
            "neighbor_bivector_dissimilarity": dissimilarity,
            "probe": probe,
            "init_mode": init_mode,
        }

    def _map_bivector_parameters(
        self,
        probe: _SignatureProbe,
        algebra: AlgebraContext,
        original_dim: int,
    ) -> Tuple[Tuple[int, int], Dict]:
        """Map bivector parameter energies to heuristic active counts ``(p, q)``."""
        bv_sq = build_bivector_squared_signs(algebra.layout((2,)), device=self.device, dtype=algebra.dtype)
        bv_indices = algebra.layout((2,)).indices_tensor(device=self.device)

        total_energy = probe.rotor.bivector_parameters.detach().square().mean(dim=0)

        max_energy = total_energy.max().clamp(min=algebra.eps)
        normalized_energy = total_energy / max_energy

        n = algebra.n
        base_type_energy: dict = {}
        base_active: dict = {}

        for bv_idx_pos, blade_idx in enumerate(bv_indices.tolist()):
            energy_val = normalized_energy[bv_idx_pos].item()
            bits = []
            for bit in range(n):
                if blade_idx & (1 << bit):
                    bits.append(bit)
            if len(bits) != 2:
                continue

            sq_val = bv_sq[bv_idx_pos].item()
            if sq_val < _BV_SQ_ELLIPTIC_BOUND:
                sig_type = "elliptic"
            elif sq_val > _BV_SQ_HYPERBOLIC_BOUND:
                sig_type = "hyperbolic"
            else:
                raise ValueError("signature probing requires a nondegenerate host")

            for b in bits:
                if b not in base_type_energy:
                    base_type_energy[b] = {
                        "elliptic": 0.0,
                        "hyperbolic": 0.0,
                    }
                    base_active[b] = 0.0
                base_type_energy[b][sig_type] += energy_val
                base_active[b] = max(base_active[b], energy_val)

        elliptic_dominant_count = 0
        hyperbolic_dominant_count = 0

        for b_idx in range(n):
            if (
                b_idx not in base_active
                or base_active[b_idx] <= 0
                or base_active[b_idx] < self.bivector_parameter_energy_threshold
            ):
                continue
            type_energy = base_type_energy[b_idx]
            dominant = max(type_energy, key=type_energy.get)
            if dominant == "hyperbolic":
                hyperbolic_dominant_count += 1
            else:
                elliptic_dominant_count += 1

        p = max(0, elliptic_dominant_count - 1)
        q = max(0, hyperbolic_dominant_count - 1)

        total = p + q
        if total > original_dim:
            scale = original_dim / max(total, 1)
            p = round(p * scale)
            q = round(q * scale)
            while p + q > original_dim:
                if q > 0:
                    q -= 1
                else:
                    p -= 1

        bivector_parameter_summary = {
            "normalized_bivector_parameter_energy": normalized_energy.tolist(),
            "elliptic_dominant_count": elliptic_dominant_count,
            "hyperbolic_dominant_count": hyperbolic_dominant_count,
            "bivector_square_signs": bv_sq.tolist(),
        }

        return (p, q), bivector_parameter_summary

    def _run_probes(self, data: torch.Tensor) -> Dict:
        """Return the candidate tuple and probe diagnostics.

        Args:
            data (torch.Tensor): Input data [N, D].

        Returns:
            Diagnostics with ``candidate_active_pq``, neighborhood bivector
            scores, bivector parameter summary, and per-probe results.
        """
        data = data.to(self.device)
        N, X = data.shape

        mv_data, algebra = self._quadratic_lift(data)

        init_modes = ["elliptic_weighted", "non_null_weighted", "uniform"]
        while len(init_modes) < self.num_probes:
            init_modes.append("normal")
        init_modes = init_modes[: self.num_probes]

        def _run_probe(init_mode):
            return self._train_probe(mv_data, algebra, init_mode)

        if self.num_probes <= 2:
            probe_results = [_run_probe(bt) for bt in init_modes]
        else:
            max_w = min(self.num_probes, _SIGNATURE_PROBE_PARALLEL_WORKER_CAP)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = [pool.submit(_run_probe, bt) for bt in init_modes]
                probe_results = [f.result() for f in futures]

        best_idx = min(range(len(probe_results)), key=lambda i: probe_results[i]["best_training_loss"])
        best = probe_results[best_idx]

        candidate_active_pq, bivector_parameter_summary = self._map_bivector_parameters(best["probe"], algebra, X)

        return {
            "candidate_active_pq": candidate_active_pq,
            "unresolved_features": X - sum(candidate_active_pq),
            "neighbor_bivector_alignment": best["neighbor_bivector_alignment"],
            "neighbor_bivector_dissimilarity": best["neighbor_bivector_dissimilarity"],
            "bivector_parameter_summary": bivector_parameter_summary,
            "per_probe_results": [
                {
                    "init_mode": r["init_mode"],
                    "best_training_loss": r["best_training_loss"],
                    "neighbor_bivector_alignment": r["neighbor_bivector_alignment"],
                    "neighbor_bivector_dissimilarity": r["neighbor_bivector_dissimilarity"],
                }
                for r in probe_results
            ],
        }
