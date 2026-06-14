# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""High-dimensional lattice-basis orthogonalization with injected criteria."""

from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass

import torch

if __package__ in {None, ""}:
    from _common import add_runtime_arguments, bootstrap_repo_root, configure_runtime, print_latest, resolve_runtime
else:
    from ._common import add_runtime_arguments, bootstrap_repo_root, configure_runtime, print_latest, resolve_runtime

bootstrap_repo_root(__file__)

from clifra.core.runtime.algebra import AlgebraContext
from research.continuum_solver import (
    BivectorNormPolicy,
    ContinuumSolverEngine,
    CriterionResult,
    InvertibleBivectorField,
    InvertiblePathConsistencyPolicy,
    PolicyResult,
)


@dataclass(frozen=True)
class OrthogonalLatticeBasisCriterion:
    """Make a twisted lattice basis angularly orthogonal."""

    source_basis: torch.Tensor
    source_active: torch.Tensor
    active_subspace: torch.Tensor
    log_condition_weight: float = 0.03
    name: str = "orthogonal_lattice_basis"

    def __call__(self, engine, state) -> CriterionResult:
        full_basis = reconstruct_basis(
            state.deformed_coordinates,
            source_basis=self.source_basis,
            source_active=self.source_active,
            active_subspace=self.active_subspace,
        )
        corr, offdiag, eigenvalues = _correlation_gram(full_basis)
        offdiag_mse = offdiag.square().mean()
        min_eig = eigenvalues.amin().clamp_min(1e-12)
        max_eig = eigenvalues.amax().clamp_min(1e-12)
        log10_condition = torch.log10(max_eig / min_eig)
        loss = offdiag_mse + float(self.log_condition_weight) * log10_condition.square()
        return CriterionResult(
            name=self.name,
            loss=loss,
            metrics={
                "offdiag_rms": offdiag_mse.sqrt(),
                "max_abs_correlation": offdiag.abs().amax(),
                "log10_condition": log10_condition,
                "min_correlation_eigenvalue": min_eig,
                "log10_lattice_volume": _log10_volume(full_basis),
            },
        )


@dataclass(frozen=True)
class ContinuousDifferentialFlowPolicy:
    """Keep neighboring basis-vector flows smooth along the lattice index."""

    generator_weight: float = 0.05
    deformation_weight: float = 0.25
    weight: float = 1.0
    strict_tolerance: float = 5e-3
    name: str = "continuous_differential_flow"

    def __call__(self, engine, state) -> PolicyResult:
        weights = state.bivector_weights
        if weights.shape[1] > 1:
            generator_delta = weights.diff(dim=1)
            generator_smoothness = generator_delta.square().mean()
            max_generator_step = torch.linalg.vector_norm(generator_delta, dim=-1).amax()
        else:
            generator_smoothness = weights.new_zeros(())
            max_generator_step = weights.new_zeros(())

        displacement = state.deformed_coordinates - state.reference_coordinates
        if displacement.shape[0] > 2:
            second_difference = displacement.diff(dim=0).diff(dim=0)
            deformation_smoothness = second_difference.square().mean()
            max_second_difference = torch.linalg.vector_norm(second_difference, dim=-1).amax()
        else:
            deformation_smoothness = displacement.new_zeros(())
            max_second_difference = displacement.new_zeros(())

        loss = float(self.generator_weight) * generator_smoothness + float(self.deformation_weight) * deformation_smoothness
        max_violation = torch.maximum(max_generator_step, max_second_difference)
        return PolicyResult(
            name=self.name,
            loss=loss,
            weight=self.weight,
            strict_tolerance=self.strict_tolerance,
            metrics={
                "generator_smoothness": generator_smoothness,
                "deformation_smoothness": deformation_smoothness,
                "max_generator_step": max_generator_step,
                "max_second_difference": max_second_difference,
            },
            violations={"max_flow_jump": max_violation},
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--active-dim", type=int, default=12)
    parser.add_argument("--path-steps", type=int, default=2)
    parser.add_argument("--control-points", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--condition-log10", type=float, default=5.0)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    runtime = resolve_runtime(args)
    configure_runtime(seed=43, max_threads=args.max_threads)

    dim = int(args.dim)
    active_dim = int(args.active_dim)
    if active_dim <= 1 or active_dim > dim:
        raise ValueError(f"active_dim must be in [2, {dim}], got {active_dim}")
    source_basis = twisted_ill_conditioned_basis(
        dim,
        log10_condition=float(args.condition_log10),
        device=runtime.device,
        dtype=runtime.dtype,
    )
    print_basis_metrics("Initial 48D Lattice" if dim == 48 else f"Initial {dim}D Lattice", source_basis)
    active_subspace = _orthogonal_matrix(dim, seed=307, dtype=runtime.dtype, device=runtime.device)[:, :active_dim]
    source_active = source_basis @ active_subspace

    algebra = AlgebraContext(p=active_dim, q=0, r=0, device=runtime.device, dtype=runtime.dtype)
    field = InvertibleBivectorField(
        algebra,
        coordinate_dim=active_dim,
        path_steps=args.path_steps,
        control_shape=(min(args.control_points, active_dim),),
        init_scale=1e-2,
    )
    engine = ContinuumSolverEngine(
        field,
        target_criterion=OrthogonalLatticeBasisCriterion(
            source_basis=source_basis,
            source_active=source_active,
            active_subspace=active_subspace,
        ),
        geometric_policies=(
            ContinuousDifferentialFlowPolicy(),
            InvertiblePathConsistencyPolicy(weight=0.5),
            BivectorNormPolicy(max_norm=1.8, weight=1e-3),
        ),
    )

    run = engine.fit(
        source_active,
        steps=args.steps,
        lr=args.lr,
        log_every=args.log_every or max(1, args.steps // 10),
        clip_grad_norm=args.clip_grad_norm,
        compile_step=runtime.compile_step,
        compile_backend=runtime.compile_backend,
        compile_mode=runtime.compile_mode,
        compile_fullgraph=runtime.compile_fullgraph,
    )
    print_latest(run.history, title=f"{dim}D Lattice Basis Flow ({active_dim}D reversible subspace)", limit=32)
    final_basis = reconstruct_basis(run.output, source_basis=source_basis, source_active=source_active, active_subspace=active_subspace)
    print_basis_metrics("Final Lattice", final_basis)


def twisted_ill_conditioned_basis(dim: int, *, log10_condition: float, device, dtype: torch.dtype) -> torch.Tensor:
    """Return a deterministic row-normalized but angularly ill-conditioned basis."""
    left = _orthogonal_matrix(dim, seed=101, dtype=dtype, device=device)
    right = _orthogonal_matrix(dim, seed=211, dtype=dtype, device=device)
    exponents = torch.linspace(-0.5 * log10_condition, 0.5 * log10_condition, dim, device=device, dtype=dtype)
    singular_values = torch.pow(torch.tensor(10.0, device=device, dtype=dtype), exponents)
    basis = left @ torch.diag(singular_values) @ right.transpose(-2, -1)
    basis = basis + 0.08 * torch.roll(basis, shifts=1, dims=1) - 0.05 * torch.roll(basis, shifts=3, dims=0)
    return basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True).clamp_min(1e-12)


def reconstruct_basis(
    active_coordinates: torch.Tensor,
    *,
    source_basis: torch.Tensor,
    source_active: torch.Tensor,
    active_subspace: torch.Tensor,
) -> torch.Tensor:
    """Lift reversible active-subspace coordinates back into the full lattice basis."""
    source_basis = source_basis.to(device=active_coordinates.device, dtype=active_coordinates.dtype)
    source_active = source_active.to(device=active_coordinates.device, dtype=active_coordinates.dtype)
    active_subspace = active_subspace.to(device=active_coordinates.device, dtype=active_coordinates.dtype)
    return source_basis + (active_coordinates - source_active) @ active_subspace.transpose(-2, -1)


def print_basis_metrics(title: str, basis: torch.Tensor) -> None:
    corr, offdiag, eigenvalues = _correlation_gram(basis)
    print(f"\n== {title} ==")
    print(f"offdiag_rms: {_format_scalar(offdiag.square().mean().sqrt())}")
    print(f"max_abs_correlation: {_format_scalar(offdiag.abs().amax())}")
    print(f"log10_condition: {_format_scalar(torch.log10(eigenvalues.amax() / eigenvalues.amin().clamp_min(1e-12)))}")
    print(f"log10_lattice_volume: {_format_scalar(_log10_volume(basis))}")


def _orthogonal_matrix(dim: int, *, seed: int, dtype: torch.dtype, device) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    matrix = torch.randn(dim, dim, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (q * signs.unsqueeze(0)).to(device=device)


def _correlation_gram(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gram = basis @ basis.transpose(-2, -1)
    diagonal = torch.diagonal(gram).clamp_min(1e-12)
    normalizer = torch.sqrt(diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).clamp_min(1e-12)
    corr = gram / normalizer
    eye = torch.eye(corr.shape[-1], device=corr.device, dtype=corr.dtype)
    offdiag = corr - eye
    eigenvalues = torch.linalg.eigvalsh(corr + 1e-10 * eye).clamp_min(1e-12)
    return corr, offdiag, eigenvalues


def _log10_volume(basis: torch.Tensor) -> torch.Tensor:
    gram = basis @ basis.transpose(-2, -1)
    eps = torch.finfo(gram.dtype).eps
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(eps)
    return 0.5 * torch.log10(eigenvalues).sum(dim=-1)


def _format_scalar(value: torch.Tensor) -> str:
    return f"{float(value.detach().cpu().reshape(())):.6g}"


if __name__ == "__main__":
    main()
