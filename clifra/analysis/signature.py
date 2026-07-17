# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Experimental rotor-probe metric-signature estimation.

Provides :class:`RotorProbeSignatureEstimator` (a learned heuristic) and
:class:`SignatureProbeAnalyzer` (higher-level wrapper with dimension
reduction and bootstrap agreement estimates).

These routines select a candidate signature from trained probe energy. They
do not identify or prove the metric signature of the source data.
"""

import concurrent.futures
import copy
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from clifra.core.config import make_algebra
from clifra.core.execution.action import FullSandwichActionExecutor, full_versor_factors
from clifra.core.foundation.layout import AlgebraSpec
from clifra.core.foundation.module import AlgebraLike, CliffordModule

from ._types import CONSTANTS, DimensionResult, SamplingConfig, SignatureEstimate
from ._utils import action_matrix_feasibility_for_spec, analysis_dtype, as_analysis_tensor
from .geodesic import NeighborhoodBivectorFlow


class _ProbeLinear(CliffordModule):
    """Core-local channel mixer used by metric-search probes."""

    def __init__(self, algebra: AlgebraLike, in_channels: int, out_channels: int):
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
    """Core-local full-lane rotor used only for signature search analysis."""

    def __init__(self, algebra: AlgebraLike, channels: int):
        super().__init__(algebra)
        self.channels = channels
        self.parameter_layout = algebra.layout((2,))
        self.full_layout = algebra.layout()
        self.action = FullSandwichActionExecutor.from_layout(
            self.full_layout,
            device=algebra.device,
            dtype=algebra.dtype,
        )
        self.bivector_weights = nn.Parameter(torch.empty(channels, self.parameter_layout.dim))
        self.bivector_weights._manifold = "spin"
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.bivector_weights, std=CONSTANTS.signature_probe_rotor_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = full_versor_factors(
            self.algebra,
            self.bivector_weights.to(device=x.device, dtype=x.dtype),
            grade=2,
            parameter_layout=self.parameter_layout,
        )
        return self.action.per_channel(left, x, right)

    def sparsity_loss(self) -> torch.Tensor:
        return torch.norm(self.bivector_weights, p=1)


class _ProbeBladeSelector(CliffordModule):
    """Core-local blade gate for metric-search probes."""

    def __init__(self, algebra: AlgebraLike, channels: int):
        super().__init__(algebra)
        self.weights = nn.Parameter(torch.ones(channels, algebra.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.weights).unsqueeze(0)


class _SignatureProbe(nn.Module):
    """Minimal single-rotor probe for bivector energy analysis.

    Architecture: channel mixer -> rotor -> channel mixer -> blade selector.
    Only one linear layer for channel expansion; the rotor bivector energy
    is the signal used by the experimental signature heuristic.
    """

    def __init__(self, algebra: AlgebraLike, channels: int = CONSTANTS.signature_probe_channels):
        super().__init__()
        self.algebra = algebra
        self.linear_in = _ProbeLinear(algebra, 1, channels)
        self.rotor = _ProbeRotor(algebra, channels)
        self.linear_out = _ProbeLinear(algebra, channels, 1)
        self.selector = _ProbeBladeSelector(algebra, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.rotor(x)
        x = self.linear_out(x)
        x = self.selector(x)
        return x

    def get_rotor_layers(self) -> List[_ProbeRotor]:
        return [m for m in self.modules() if isinstance(m, _ProbeRotor)]


def _apply_biased_init(
    probe: _SignatureProbe,
    algebra: AlgebraLike,
    bias_type: str = "random",
) -> None:
    """Bias probe bivector weights based on signature type.

    Uses ``algebra.bivector_squared_signs()`` to classify each basis bivector:
    - bv_sq = -1: elliptic (positive-signature base vectors)
    - bv_sq = +1: hyperbolic (mixed-signature base vectors)
    - bv_sq =  0: null (degenerate base vectors)
    """
    bv_sq = algebra.bivector_squared_signs(device=algebra.device, dtype=algebra.dtype)
    ell = CONSTANTS.bv_sq_elliptic_bound
    hyp = CONSTANTS.bv_sq_hyperbolic_bound
    for rotor in probe.get_rotor_layers():
        with torch.no_grad():
            if bias_type == "euclidean":
                weights = torch.where(
                    bv_sq < ell,
                    torch.ones_like(bv_sq),
                    torch.full_like(bv_sq, CONSTANTS.signature_probe_bias_minor_weight),
                )
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * CONSTANTS.signature_probe_bias_noise_std
                )
            elif bias_type == "minkowski":
                weights = torch.where(
                    bv_sq.abs() > hyp,
                    torch.ones_like(bv_sq),
                    torch.full_like(bv_sq, CONSTANTS.signature_probe_bias_minor_weight),
                )
                rotor.bivector_weights.copy_(
                    weights.unsqueeze(0).expand_as(rotor.bivector_weights)
                    + torch.randn_like(rotor.bivector_weights) * CONSTANTS.signature_probe_bias_noise_std
                )
            elif bias_type == "projective":
                nn.init.uniform_(
                    rotor.bivector_weights,
                    -CONSTANTS.signature_probe_projective_init_bound,
                    CONSTANTS.signature_probe_projective_init_bound,
                )
            else:  # 'random'
                nn.init.normal_(rotor.bivector_weights, 0.0, CONSTANTS.signature_probe_random_init_std)


class RotorProbeSignatureEstimator:
    """Experimentally select a ``(p, q, r)`` candidate from rotor probes.

    Trains small single-rotor probes on conformally-lifted data using
    connection_alignment + connection_dissimilarity as the loss. After training, reads the learned
    bivector energy distribution to select a signature estimate.

    Multiple biased initializations reduce sensitivity to local minima. The
    returned tuple is a model-dependent estimate, not metric identification.
    """

    def __init__(
        self,
        device: str = "cpu",
        num_probes: int = CONSTANTS.signature_probe_num_probes,
        probe_epochs: int = CONSTANTS.signature_probe_epochs,
        probe_lr: float = CONSTANTS.signature_probe_lr,
        probe_channels: int = CONSTANTS.signature_probe_channels,
        k: int = CONSTANTS.default_k_neighbors,
        energy_threshold: float = CONSTANTS.default_energy_threshold,
        connection_dissimilarity_weight: float = CONSTANTS.signature_probe_connection_dissimilarity_weight,
        sparsity_weight: float = CONSTANTS.signature_probe_sparsity_weight,
        max_workers: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        early_stop_patience: int = 0,
        dtype: torch.dtype = CONSTANTS.default_dtype,
    ):
        self.device = device
        self.dtype = analysis_dtype(dtype)
        self.num_probes = num_probes
        self.probe_epochs = probe_epochs
        self.probe_lr = probe_lr
        self.probe_channels = probe_channels
        self.k = k
        self.energy_threshold = energy_threshold
        self.connection_dissimilarity_weight = connection_dissimilarity_weight
        self.sparsity_weight = sparsity_weight
        self.max_workers = max_workers
        self.micro_batch_size = micro_batch_size
        self.early_stop_patience = early_stop_patience

    def _lift_data(self, data: torch.Tensor) -> Tuple[torch.Tensor, AlgebraLike]:
        """Lifts [N, X] data into Cl(X+1, 1, 0) via CGA-style embedding."""
        data = as_analysis_tensor(data, device=self.device, dtype=self.dtype)
        N, X = data.shape

        spec = AlgebraSpec(X + 1, 1, 0)
        action_feasible = action_matrix_feasibility_for_spec(
            spec,
            role="signature_probe",
            max_entries=CONSTANTS.signature_probe_action_matrix_entries,
        )
        max_probe_n = CONSTANTS.signature_probe_action_matrix_lanes.bit_length() - 1
        max_input_dim = max_probe_n - CONSTANTS.signature_probe_cga_extra_dims
        if not action_feasible:
            entries = action_feasible.details["matrix_entries"]
            max_entries = action_feasible.details["max_entries"]
            warnings.warn(
                f"Data dimension {X} yields conformal probe full lanes {action_feasible.details['full_lanes']} "
                f"and action-matrix entries {entries}, exceeding max_entries={max_entries}. "
                "RotorProbeSignatureEstimator probes require full-lane rotor actions; use SignatureProbeAnalyzer PCA "
                f"pre-reduction to X <= {max_input_dim}."
            )
            raise ValueError(
                f"RotorProbeSignatureEstimator requires X <= {max_input_dim} so the conformal probe action matrix stays within "
                f"{max_entries} entries."
            )

        half_radius_squared = 0.5 * (data**2).sum(dim=-1, keepdim=True)
        ones = torch.ones(N, 1, device=self.device, dtype=data.dtype)
        lifted = torch.cat([data, half_radius_squared, ones], dim=-1)

        algebra = make_algebra(X + 1, 1, 0, device=self.device, dtype=data.dtype)
        mv = algebra.embed_vector(lifted)
        mv = mv.unsqueeze(1)
        return mv, algebra

    def _train_probe(
        self,
        mv_data: torch.Tensor,
        algebra: AlgebraLike,
        bias_type: str = "random",
    ) -> Dict:
        """Trains a single probe and returns results."""
        probe = _SignatureProbe(algebra, channels=self.probe_channels)
        probe.to(self.device)
        _apply_biased_init(probe, algebra, bias_type)

        gf = NeighborhoodBivectorFlow(algebra, k=self.k)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.probe_lr)

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        N = mv_data.shape[0]

        for _ in range(self.probe_epochs):
            if self.micro_batch_size and self.micro_batch_size < N:
                idx = torch.randperm(N, device=mv_data.device)[: self.micro_batch_size]
                batch = mv_data[idx]
            else:
                batch = mv_data

            optimizer.zero_grad()
            output = probe(batch)
            output_flat = output.squeeze(1)

            connection_alignment_t = gf._connection_alignment_tensor(output_flat)
            connection_dissimilarity_t = gf._connection_dissimilarity_tensor(output_flat)

            sparsity = sum(r.sparsity_loss() for r in probe.get_rotor_layers())

            loss = (
                -connection_alignment_t
                + self.connection_dissimilarity_weight * connection_dissimilarity_t
                + self.sparsity_weight * sparsity
            )

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val
                best_state = copy.deepcopy(probe.state_dict())
                patience_counter = 0
            elif self.early_stop_patience > 0:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    break

        if best_state is not None:
            probe.load_state_dict(best_state)

        with torch.no_grad():
            output = probe(mv_data).squeeze(1)
            alignment = gf.connection_alignment(output)
            dissimilarity = gf.connection_dissimilarity(output)

        return {
            "loss": best_loss,
            "connection_alignment": alignment,
            "connection_dissimilarity": dissimilarity,
            "probe": probe,
        }

    def _analyze_bivector_energy(
        self,
        probe: _SignatureProbe,
        algebra: AlgebraLike,
        original_dim: int,
    ) -> Tuple[Tuple[int, int, int], Dict]:
        """Map learned bivector energy to a candidate ``(p, q, r)`` tuple."""
        bv_sq = algebra.bivector_squared_signs(device=self.device, dtype=algebra.dtype)
        bv_indices = algebra.grade_indices((2,), device=self.device)

        total_energy = torch.zeros(len(bv_indices), device=self.device)
        n_layers = 0
        for rotor in probe.get_rotor_layers():
            with torch.no_grad():
                energy = (rotor.bivector_weights**2).mean(dim=0)
                total_energy += energy
                n_layers += 1

        if n_layers > 0:
            total_energy /= n_layers

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
            if sq_val < CONSTANTS.bv_sq_elliptic_bound:
                sig_type = "elliptic"
            elif sq_val > CONSTANTS.bv_sq_hyperbolic_bound:
                sig_type = "hyperbolic"
            else:
                sig_type = "null"

            for b in bits:
                if b not in base_type_energy:
                    base_type_energy[b] = {
                        "elliptic": 0.0,
                        "hyperbolic": 0.0,
                        "null": 0.0,
                    }
                    base_active[b] = 0.0
                base_type_energy[b][sig_type] += energy_val
                base_active[b] = max(base_active[b], energy_val)

        active_positive = 0
        active_negative = 0
        active_null = 0

        for b_idx in range(n):
            if b_idx not in base_active or base_active[b_idx] < self.energy_threshold:
                continue
            type_energy = base_type_energy[b_idx]
            dominant = max(type_energy, key=type_energy.get)
            if dominant == "null":
                active_null += 1
            elif dominant == "hyperbolic":
                active_negative += 1
            else:
                active_positive += 1

        p = max(0, active_positive - 1)
        q = max(0, active_negative - 1)
        r = active_null

        total = p + q + r
        if total > original_dim:
            scale = original_dim / max(total, 1)
            p = max(1, round(p * scale))
            q = round(q * scale)
            r = round(r * scale)
            while p + q + r > original_dim:
                if r > 0:
                    r -= 1
                elif q > 0:
                    q -= 1
                else:
                    p -= 1
        elif total == 0:
            p = original_dim

        energy_breakdown = {
            "per_bivector_energy": normalized_energy.tolist(),
            "active_positive": active_positive,
            "active_negative": active_negative,
            "active_null": active_null,
            "bivector_squared_signs": bv_sq.tolist(),
        }

        return (p, q, r), energy_breakdown

    def estimate(self, data: torch.Tensor) -> Tuple[int, int, int]:
        """Return the selected ``(p, q, r)`` signature estimate.

        Args:
            data (torch.Tensor): Input data [N, D].

        Returns:
            Tuple[int, int, int]: Selected signature estimate.
        """
        result = self.estimate_detailed(data)
        return result["estimated_signature"]

    def estimate_detailed(self, data: torch.Tensor) -> Dict:
        """Return the candidate tuple and probe diagnostics.

        Args:
            data (torch.Tensor): Input data [N, D].

        Returns:
            Diagnostics with ``estimated_signature``, operational connection
            scores, energy breakdown, and per-probe results.
        """
        data = data.to(self.device)
        N, X = data.shape

        mv_data, algebra = self._lift_data(data)

        bias_types = ["euclidean", "minkowski", "projective"]
        while len(bias_types) < self.num_probes:
            bias_types.append("random")
        bias_types = bias_types[: self.num_probes]

        def _run_probe(bias_type):
            return self._train_probe(mv_data, algebra, bias_type)

        if self.num_probes <= 2:
            probe_results = [_run_probe(bt) for bt in bias_types]
        else:
            max_w = self.max_workers or min(self.num_probes, CONSTANTS.signature_probe_parallel_worker_cap)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_w) as pool:
                futures = [pool.submit(_run_probe, bt) for bt in bias_types]
                probe_results = [f.result() for f in futures]

        best_idx = min(range(len(probe_results)), key=lambda i: probe_results[i]["loss"])
        best = probe_results[best_idx]

        estimated_signature, energy_breakdown = self._analyze_bivector_energy(best["probe"], algebra, X)

        return {
            "estimated_signature": estimated_signature,
            "connection_alignment": best["connection_alignment"],
            "connection_dissimilarity": best["connection_dissimilarity"],
            "energy_breakdown": energy_breakdown,
            "per_probe_results": [
                {
                    "loss": r["loss"],
                    "connection_alignment": r["connection_alignment"],
                    "connection_dissimilarity": r["connection_dissimilarity"],
                }
                for r in probe_results
            ],
        }


class SignatureProbeAnalyzer:
    """Experimental signature-candidate analysis with optional PCA reduction.

    Wraps :class:`RotorProbeSignatureEstimator` and adds:

    * Automatic PCA reduction when the input exceeds the algebra
      tractability threshold.
    * Bootstrap vote distribution and majority agreement.

    Args:
        device: Torch device string.
        max_search_dim: Maximum data dimension before PCA reduction.
            CGA lift adds 2, so ``max_search_dim=10`` -> Cl(11,1) with
            2^12 = 4096-dim multivectors (the algebra ceiling).
            Defaults to 10.
        signature_probe_kwargs: Extra keyword arguments forwarded to
            :class:`RotorProbeSignatureEstimator`.
    """

    def __init__(
        self,
        device: str = "cpu",
        max_search_dim: int = CONSTANTS.signature_search_max_dim,
        signature_probe_kwargs: Optional[Dict] = None,
        dtype: torch.dtype = CONSTANTS.default_dtype,
    ):
        self.device = device
        self.max_search_dim = max_search_dim
        self.dtype = analysis_dtype(dtype)
        self._estimator_kwargs = signature_probe_kwargs or {}

    def analyze(
        self,
        data: torch.Tensor,
        dim_result: Optional[DimensionResult] = None,
    ) -> SignatureEstimate:
        """Run the rotor-probe heuristic, optionally reducing dimensions.

        Args:
            data: ``[N, D]`` raw data.
            dim_result: Pre-computed dimension analysis.  When provided
                and ``dim_result.broken_stick_dimension < D``, the data is
                PCA-reduced before searching.

        Returns:
            :class:`SignatureEstimate`.
        """
        from .dimension import CovarianceDimensionAnalyzer

        data = as_analysis_tensor(data, device=self.device, dtype=self.dtype)
        input_dimension_used = None

        target_dim = data.shape[1]
        if dim_result is not None and dim_result.broken_stick_dimension < target_dim:
            target_dim = dim_result.broken_stick_dimension

        if target_dim > self.max_search_dim:
            target_dim = self.max_search_dim

        if target_dim < data.shape[1]:
            reducer = CovarianceDimensionAnalyzer(device=self.device, dtype=data.dtype)
            data = reducer.reduce(data, target_dim)
            input_dimension_used = target_dim

        estimator_kwargs = {"dtype": data.dtype, **self._estimator_kwargs}
        estimator = RotorProbeSignatureEstimator(device=self.device, **estimator_kwargs)
        result = estimator.estimate_detailed(data)

        return SignatureEstimate(
            estimated_signature=result["estimated_signature"],
            connection_alignment=result.get("connection_alignment", 0.0),
            connection_dissimilarity=result.get("connection_dissimilarity", 0.0),
            energy_breakdown=result.get("energy_breakdown", {}),
            input_dimension_used=input_dimension_used,
        )

    def analyze_with_confidence(
        self,
        data: torch.Tensor,
        n_bootstrap: int = CONSTANTS.signature_bootstrap_resamples,
        dim_result: Optional[DimensionResult] = None,
    ) -> Tuple[SignatureEstimate, Dict]:
        """Run the signature heuristic on bootstrap resamples.

        Runs the search on *n_bootstrap* resampled datasets and reports
        the majority-vote signature plus the full distribution.

        Returns:
            ``(best_result, confidence)`` where *confidence* contains
            ``"distribution"`` (Counter) and ``"agreement"`` (float).
        """
        from .sampler import StatisticalSampler

        cfg = SamplingConfig(
            strategy="bootstrap",
            max_samples=min(data.shape[0], CONSTANTS.signature_bootstrap_max_samples),
            n_bootstrap=n_bootstrap,
        )
        resamples, _ = StatisticalSampler.sample(data, cfg)

        signatures = []
        results = []
        for sample in resamples:
            r = self.analyze(sample, dim_result=dim_result)
            signatures.append(r.estimated_signature)
            results.append(r)

        counter = Counter(signatures)
        best_sig = counter.most_common(1)[0][0]
        agreement = counter[best_sig] / len(signatures)

        best_result = next(r for r in results if r.estimated_signature == best_sig)

        confidence = {
            "distribution": dict(counter),
            "agreement": agreement,
            "n_bootstrap": n_bootstrap,
        }
        return best_result, confidence
