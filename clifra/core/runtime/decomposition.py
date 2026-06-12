# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0


"""Bivector decomposition via GA power iteration.

Decomposes a general bivector into simple (blade) components that can each
be exponentiated with the closed-form formula.

Reference:
    Pence, T., Yamada, D., & Singh, V. (2025). "Composing Linear Layers
    from Irreducibles." arXiv:2507.11688v1 [cs.LG]
"""

from typing import List, Optional, Tuple

import torch


def _full_layout(algebra):
    """Return the canonical full layout for explicit planned decomposition calls."""
    return algebra.layout(tuple(range(int(algebra.n) + 1)))


def _seed_vector(algebra, b: torch.Tensor) -> torch.Tensor:
    """Deterministic grade-1 seed for power iteration.

    Probes ``b`` with a uniform unit vector ``(1/sqrt(n)) * sum_i e_i`` via
    right-contraction. The probe lives in ``b``'s column space, so it has
    non-zero overlap with the dominant eigenvector unless that eigenvector
    is exactly orthogonal to ``(1, ..., 1)`` -- a measure-zero case for
    which we fall back to the uniform vector itself.
    """
    batch_shape = b.shape[:-1]
    device, dtype = b.device, b.dtype
    n = algebra.n

    uniform = torch.full((*batch_shape, n), 1.0 / (n**0.5), device=device, dtype=dtype)
    v_uniform = algebra.embed_vector(uniform)

    full_layout = _full_layout(algebra)
    probe = algebra.right_contraction(
        b,
        v_uniform,
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
    )
    probe_norm = probe.norm(dim=-1, keepdim=True)
    return torch.where(probe_norm > algebra.eps, probe, v_uniform)


def ga_power_iteration(
    algebra, b: torch.Tensor, v_init: Optional[torch.Tensor] = None, threshold: float = 1e-6, max_iterations: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the dominant simple bivector component via power iteration.

    Implements Algorithm 2 from Pence et al. (2025).  Iterates
    ``v <- (b _| v) / ||b _| v||`` until convergence, then recovers
    the simple projection ``b_s = sigma * (u ^ v)``.

    Args:
        algebra: Layout-first algebra context.
        b: Bivector to decompose [..., dim].
        v_init: Initial grade-1 vector (random if None).
        threshold: Convergence tolerance on ``||v - v_prev||``.
        max_iterations: Iteration cap.

    Returns:
        (b_s, v) where b_s is the simple projection and v the converged
        vector, both shaped [..., dim].
    """
    if v_init is None:
        v = _seed_vector(algebra, b)
    else:
        v = v_init
    full_layout = _full_layout(algebra)

    v_norm = v.norm(dim=-1, keepdim=True)
    v = v / v_norm.clamp(min=algebra.eps)

    for _ in range(max_iterations):
        v_prev = v
        v = algebra.right_contraction(
            b,
            v,
            left_layout=full_layout,
            right_layout=full_layout,
            output_layout=full_layout,
        )
        v_norm = v.norm(dim=-1, keepdim=True)
        v = v / v_norm.clamp(min=algebra.eps)

        if (v - v_prev).norm(dim=-1).max() < threshold:
            break

    u = algebra.right_contraction(
        b,
        v,
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
    )
    u_norm = u.norm(dim=-1, keepdim=True)
    u = u / u_norm.clamp(min=algebra.eps)

    b_s = u_norm * algebra.wedge(
        u,
        v,
        left_layout=full_layout,
        right_layout=full_layout,
        output_layout=full_layout,
    )

    return b_s, v


def differentiable_invariant_decomposition(
    algebra, b: torch.Tensor, k: Optional[int] = None, threshold: float = 1e-6, max_iterations: int = 100
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Decompose a bivector into simple components via greedy projection.

    Implements Algorithm 1 from Pence et al. (2025).  Iteratively
    extracts the dominant simple component and subtracts it from the
    residual.

    Args:
        algebra: Layout-first algebra context.
        b: Bivector [..., dim].
        k: Number of components (auto = n(n-1)/2 if None).
        threshold: Stop when residual norm falls below this.
        max_iterations: Per-component power iteration cap.

    Returns:
        (decomp, vectors): lists of simple bivectors and their
        associated vectors.
    """
    n = algebra.n
    k_max = (n * (n - 1)) // 2
    k = min(k, k_max) if k is not None else k_max

    decomp: List[torch.Tensor] = []
    vectors: List[torch.Tensor] = []
    residual = b.clone()

    for _ in range(k):
        if residual.norm(dim=-1).max() < threshold:
            break

        b_i, v_i = ga_power_iteration(algebra, residual, threshold=threshold, max_iterations=max_iterations)
        decomp.append(b_i)
        vectors.append(v_i)
        residual = residual - b_i

    return decomp, vectors
