# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Configuration defaults for the TinyStories SLM experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


def default_device() -> str:
    """Return a practical default device for single-process research runs."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype_from_name(name: str) -> torch.dtype:
    """Resolve a small stable dtype vocabulary used by CLI checkpoints."""
    key = str(name).lower().replace("torch.", "")
    if key in {"float32", "fp32"}:
        return torch.float32
    if key in {"float64", "double", "fp64"}:
        return torch.float64
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float16", "half", "fp16"}:
        return torch.float16
    raise ValueError(f"unsupported dtype {name!r}")


@dataclass(frozen=True)
class SLMConfig:
    """Decision-complete defaults for the first clifra SLM prototype."""

    p: int = 4
    q: int = 1
    r: int = 1
    even_grades: tuple[int, ...] = (0, 2, 4, 6)
    seq_len: int = 256
    vocab_size: int = 256
    channels: int = 64
    num_layers: int = 4
    num_heads: int = 4
    ffn_multiplier: int = 4
    num_ffn_versors: int = 8
    ffn_versor_chunk_channels: int | None = 32
    ffn_reflection_pairs: int = 1
    attention_query_chunk_size: int | None = 16
    attention_bivector_weight: float = 0.5
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    layer_norm_recover: bool = True
    gradient_checkpointing: bool = False
    device: str = "auto"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        n = int(self.p) + int(self.q) + int(self.r)
        expected_even = tuple(range(0, n + 1, 2))
        if tuple(self.even_grades) != expected_even:
            raise ValueError(
                f"even_grades must be the full even layout {expected_even} for Cl({self.p},{self.q},{self.r}), "
                f"got {self.even_grades}"
            )
        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1, got {self.vocab_size}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.num_heads <= 0 or self.channels % self.num_heads != 0:
            raise ValueError(f"num_heads must divide channels, got channels={self.channels}, heads={self.num_heads}")
        if self.ffn_multiplier <= 0:
            raise ValueError(f"ffn_multiplier must be positive, got {self.ffn_multiplier}")
        if self.num_ffn_versors <= 0:
            raise ValueError(f"num_ffn_versors must be positive, got {self.num_ffn_versors}")
        if self.ffn_versor_chunk_channels is not None and self.ffn_versor_chunk_channels <= 0:
            raise ValueError(f"ffn_versor_chunk_channels must be positive or None, got {self.ffn_versor_chunk_channels}")
        if self.ffn_reflection_pairs < 0:
            raise ValueError(f"ffn_reflection_pairs must be non-negative, got {self.ffn_reflection_pairs}")
        if self.attention_query_chunk_size is not None and self.attention_query_chunk_size <= 0:
            raise ValueError(f"attention_query_chunk_size must be positive or None, got {self.attention_query_chunk_size}")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        dtype_from_name(self.dtype)

    @property
    def resolved_device(self) -> str:
        """Return the concrete device string used to construct the algebra."""
        return default_device() if self.device == "auto" else self.device

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the configured torch dtype."""
        return dtype_from_name(self.dtype)

    @property
    def hidden_channels(self) -> int:
        """Return the expanded FFN channel count."""
        return self.channels * self.ffn_multiplier

    def with_overrides(self, **kwargs) -> "SLMConfig":
        """Return a validated config copy with selected fields changed."""
        return replace(self, **kwargs)
