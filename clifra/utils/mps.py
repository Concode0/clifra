# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""MPS-specific operator fallbacks."""

import torch

if torch.backends.mps.is_available():

    def safe_linalg_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Solve on CPU when the input lives on MPS."""
        if A.is_mps:
            return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
        return torch.linalg.solve(A, B)
    
    def safe_linalg_eigvals(A: torch.Tensor) -> torch.Tensor:
        if A.is_mps:
            return torch.linalg.eigvals(A.cpu()).to(A.device)
        return torch.linalg.eigvals(A)
    
    
else:
    safe_linalg_solve = torch.linalg.solve
    safe_linalg_eigvals = torch.linalg.eigvals
