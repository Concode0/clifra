# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from tests.helpers.small_oracle import SmallCliffordOracle


def bivector_exp_cpu_reference(
    algebra,
    values: torch.Tensor,
    *,
    input_layout,
    output_layout,
) -> torch.Tensor:
    oracle = SmallCliffordOracle(algebra.p, algebra.q, algebra.r)
    operator_indices = oracle.indices_for_grades(range(0, algebra.n + 1, 2))
    operator_positions = {index: position for position, index in enumerate(operator_indices)}
    cpu_values = values.to(device="cpu")
    cpu_basis = torch.eye(len(operator_indices), dtype=cpu_values.dtype, device="cpu")
    columns = oracle.product(
        cpu_values.unsqueeze(-2),
        cpu_basis,
        op="gp",
        left_indices=input_layout.basis_indices,
        right_indices=operator_indices,
        output_indices=operator_indices,
    )
    operator = columns.transpose(-1, -2)
    even_output = torch.matrix_exp(operator)[..., :, operator_positions[0]]
    output = even_output.new_zeros(*even_output.shape[:-1], output_layout.dim)
    for output_position, output_index in enumerate(output_layout.basis_indices):
        operator_position = operator_positions.get(output_index)
        if operator_position is not None:
            output[..., output_position] = even_output[..., operator_position]
    return output.to(device=values.device)
