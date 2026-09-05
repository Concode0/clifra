# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Stable callable boundary for already planned tensor operations."""

from __future__ import annotations

import copy

import torch
from torch import nn

from .tensors import TensorContract


class PlannedOperation(nn.Module):
    """Execute fixed contracts without consulting the algebra or planner.

    The concrete kernel and its buffers are private implementation details.
    Move this module with ``to`` or register it on an owning PyTorch module.
    Rebuild plans from declarations when loading across library versions.
    """

    def __init__(
        self, kernel: nn.Module, inputs: tuple[TensorContract | None, ...], output: TensorContract, *, method="forward"
    ):
        super().__init__()
        self._kernel = kernel
        self._inputs = tuple(inputs)
        self._output = output
        self._method = method

    @property
    def inputs(self) -> tuple[TensorContract | None, ...]:
        """Input declarations; None denotes an ordinary non-Clifford tensor."""
        return self._inputs

    @property
    def output(self) -> TensorContract:
        """Immutable result declaration."""
        return self._output

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        if len(values) != len(self._inputs):
            raise ValueError(f"expected {len(self._inputs)} tensor operands, got {len(values)}")
        compact = tuple(
            value if contract is None else contract.to_compact(value) for contract, value in zip(self._inputs, values)
        )
        if self._method == "forward_pairwise_compact":
            if any(value.ndim < 2 for value in compact):
                raise ValueError("pairwise products require explicit item axes")
        elif self._method == "forward_compact" and len(compact) == 2:
            for left, right in zip(reversed(compact[0].shape[:-1]), reversed(compact[1].shape[:-1])):
                if left != right and left != 1 and right != 1:
                    raise ValueError("operand batch dimensions cannot broadcast. Use pairwise=True for item axes")
        result = getattr(self._kernel, self._method)(*compact)
        if self._output.uses_canonical_storage:
            return self._output.layout.full(result)
        return result

    def _apply(self, fn):
        # Cached kernels may be shared by multiple independently owned plans.
        # Detach before movement so moving one plan cannot mutate another.
        if any(
            fn(buffer).device != buffer.device or fn(buffer).dtype != buffer.dtype for buffer in self._kernel.buffers()
        ):
            self._kernel = copy.deepcopy(self._kernel)
        return super()._apply(fn)
