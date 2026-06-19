# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Clifford TinyStories language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from clifra.core.execution.attention import GeometricAttentionScoreExecutor
from clifra.core.foundation.layout import GradeLayout
from clifra.core.foundation.module import CliffordModule
from clifra.core.runtime.algebra import AlgebraContext
from clifra.layers import CliffordLayerNorm, CliffordLinear, GeometricGELU, MultiVersorLayer
from clifra.layers.primitives.reflection import ReflectionLayer

from .config import SLMConfig


@dataclass
class SLMOutput:
    """Language-model output bundle."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


@dataclass(frozen=True)
class SLMMemoryEstimate:
    """Approximate largest activation tensors for one forward block."""

    attention_score_product_mib: float
    ffn_expanded_activation_mib: float
    ffn_multiversor_middle_mib: float
    ffn_multiversor_transformed_mib: float

    def as_dict(self) -> dict[str, float]:
        """Return estimates as plain values for logging."""
        return {
            "attention_score_product_mib": self.attention_score_product_mib,
            "ffn_expanded_activation_mib": self.ffn_expanded_activation_mib,
            "ffn_multiversor_middle_mib": self.ffn_multiversor_middle_mib,
            "ffn_multiversor_transformed_mib": self.ffn_multiversor_transformed_mib,
        }


def build_algebra(config: SLMConfig) -> AlgebraContext:
    """Build the configured Cl(4,1,1) algebra with even grades as default."""
    return AlgebraContext(
        config.p,
        config.q,
        config.r,
        device=config.resolved_device,
        dtype=config.torch_dtype,
        default_grades=config.even_grades,
    )


def estimate_activation_hotspots(config: SLMConfig, *, batch_size: int) -> SLMMemoryEstimate:
    """Estimate the main activation hot spots controlled by SLM chunking knobs."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    algebra = AlgebraContext(
        config.p,
        config.q,
        config.r,
        device="cpu",
        dtype=config.torch_dtype,
        default_grades=config.even_grades,
    )
    layout = algebra.layout(config.even_grades)
    score_layout = algebra.layout((0, 2))
    element_size = torch.empty((), dtype=config.torch_dtype).element_size()
    seq_len = int(config.seq_len)
    head_channels = config.channels // config.num_heads
    q_chunk = seq_len if config.attention_query_chunk_size is None else min(seq_len, config.attention_query_chunk_size)
    hidden_chunk = (
        config.hidden_channels
        if config.ffn_versor_chunk_channels is None
        else min(config.hidden_channels, config.ffn_versor_chunk_channels)
    )

    def mib(numel: int) -> float:
        return float(numel * element_size / (1024 * 1024))

    attention_score_product = batch_size * config.num_heads * head_channels * q_chunk * seq_len * score_layout.dim
    ffn_expanded = batch_size * seq_len * config.hidden_channels * layout.dim
    ffn_multiversor = batch_size * seq_len * config.num_ffn_versors * hidden_chunk * layout.dim
    return SLMMemoryEstimate(
        attention_score_product_mib=mib(attention_score_product),
        ffn_expanded_activation_mib=mib(ffn_expanded),
        ffn_multiversor_middle_mib=mib(ffn_multiversor),
        ffn_multiversor_transformed_mib=mib(ffn_multiversor),
    )


def _merge_attention_mask(left: torch.Tensor | None, right: torch.Tensor | None) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left | right


def _score_bias(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return scores
    return scores.masked_fill(mask, float("-inf"))


def _sdpa_from_scores(scores: torch.Tensor, values: torch.Tensor, *, dropout_p: float) -> torch.Tensor:
    B, H, Lq, Lk = scores.shape
    value_shape = values.shape
    flat_values = values.reshape(B, H, Lk, -1)
    query = scores.new_zeros(B, H, Lq, 1)
    key = scores.new_zeros(B, H, Lk, 1)
    output = F.scaled_dot_product_attention(
        query,
        key,
        flat_values,
        attn_mask=scores,
        dropout_p=dropout_p,
        scale=1.0,
    )
    return output.reshape(*value_shape[:2], Lq, *value_shape[3:])


class ChunkedGeometricProductAttention(CliffordModule):
    """Geometric attention that computes GA scores in query chunks."""

    def __init__(self, algebra: AlgebraContext, config: SLMConfig, layout: GradeLayout):
        super().__init__(algebra)
        if config.channels % config.num_heads != 0:
            raise ValueError(f"channels must divide heads, got {config.channels} and {config.num_heads}")
        self.channels = config.channels
        self.num_heads = config.num_heads
        self.head_channels = config.channels // config.num_heads
        self.causal = True
        self.query_chunk_size = config.attention_query_chunk_size
        self.layout = layout
        self.lane_dim = layout.dim
        self.qkv_weight = nn.Parameter(torch.empty(3, self.channels, self.channels))
        self.qkv_bias = nn.Parameter(torch.empty(3, self.channels, self.lane_dim))
        self.out_proj = CliffordLinear(algebra, self.channels, self.channels, backend="traditional", layout=layout)
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else None
        self.scorer = GeometricAttentionScoreExecutor(
            algebra,
            head_channels=self.head_channels,
            bivector_weight=config.attention_bivector_weight,
            layout=layout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in self.qkv_weight:
            nn.init.xavier_uniform_(weight)
        nn.init.zeros_(self.qkv_bias)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"attention input must be [B, L, C, D], got shape {tuple(x.shape)}")
        B, L, C, D = x.shape
        if C != self.channels or D != self.lane_dim:
            raise ValueError(f"attention input must have channels={self.channels}, lanes={self.lane_dim}, got {C}, {D}")

        qkv = torch.einsum("poi,...id->...pod", self.qkv_weight, x)
        qkv = qkv + self.qkv_bias.view(1, 1, 3, C, D)
        Q = qkv.select(-3, 0)
        K = qkv.select(-3, 1)
        V = qkv.select(-3, 2)

        H = self.num_heads
        Hc = self.head_channels
        Q = Q.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        K = K.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)
        V = V.reshape(B, L, H, Hc, D).permute(0, 2, 1, 3, 4)

        dropout_p = self.attn_dropout.p if self.attn_dropout is not None and self.training else 0.0
        chunk_size = L if self.query_chunk_size is None else min(L, int(self.query_chunk_size))
        padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) if key_padding_mask is not None else None
        chunks = []
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            score_mask = self._causal_mask(start, end, L, x.device) if self.causal else None
            score_mask = _merge_attention_mask(score_mask, padding_mask)
            scores = _score_bias(self.scorer(Q[:, :, start:end], K), score_mask)
            chunks.append(_sdpa_from_scores(scores, V, dropout_p=dropout_p))

        output = torch.cat(chunks, dim=2)
        output = output.permute(0, 2, 1, 3, 4).reshape(B, L, C, D)
        return self.out_proj(output)

    @staticmethod
    def _causal_mask(start: int, end: int, key_len: int, device) -> torch.Tensor:
        query_positions = torch.arange(start, end, device=device).unsqueeze(-1)
        key_positions = torch.arange(key_len, device=device).unsqueeze(0)
        return (key_positions > query_positions).unsqueeze(0).unsqueeze(0)


class EvenReflectionStack(CliffordModule):
    """Even-depth reflection stack constrained to even-layout inputs and outputs."""

    def __init__(self, algebra: AlgebraContext, channels: int, layout: GradeLayout, *, depth: int = 2):
        super().__init__(algebra)
        if depth <= 0 or depth % 2 != 0:
            raise ValueError(f"depth must be a positive even reflection count, got {depth}")
        self.layers = nn.ModuleList(
            ReflectionLayer(algebra, channels=channels, input_layout=layout, output_layout=layout)
            for _ in range(int(depth))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MultiRotorFFN(CliffordModule):
    """Traditional FFN with chunked rotor superposition and even reflection pairs."""

    def __init__(self, algebra: AlgebraContext, config: SLMConfig, layout: GradeLayout):
        super().__init__(algebra)
        hidden = config.hidden_channels
        self.versor_chunk_channels = config.ffn_versor_chunk_channels
        self.expand = CliffordLinear(algebra, config.channels, hidden, backend="traditional", layout=layout)
        self.activation = GeometricGELU(algebra, hidden, layout=layout)
        self.multi = MultiVersorLayer(
            algebra,
            channels=hidden,
            num_versors=config.num_ffn_versors,
            grade=2,
            input_layout=layout,
            output_layout=layout,
        )
        self.reflections = (
            EvenReflectionStack(algebra, channels=hidden, layout=layout, depth=2 * config.ffn_reflection_pairs)
            if config.ffn_reflection_pairs > 0
            else nn.Identity()
        )
        self.project = CliffordLinear(algebra, hidden, config.channels, backend="traditional", layout=layout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.activation(x)
        x = self._multi_versor(x)
        x = self.reflections(x)
        x = self.activation(x)
        x = self.project(x)
        return self.dropout(x)

    def _multi_versor(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[-2]
        chunk_size = channels if self.versor_chunk_channels is None else min(channels, int(self.versor_chunk_channels))
        if chunk_size >= channels:
            return self.multi(x)
        chunks = []
        for start in range(0, channels, chunk_size):
            end = min(start + chunk_size, channels)
            chunk = x[..., start:end, :]
            mix = self.multi.weights[start:end]
            chunks.append(self.multi.action(chunk, self.multi.grade_weights, mix))
        return torch.cat(chunks, dim=-2)


class SLMBlock(CliffordModule):
    """Pre-norm causal transformer block over compact even multivectors."""

    def __init__(self, algebra: AlgebraContext, config: SLMConfig, layout: GradeLayout):
        super().__init__(algebra)
        self.attn_norm = CliffordLayerNorm(
            algebra,
            config.channels,
            eps=config.layer_norm_eps,
            recover=config.layer_norm_recover,
            layout=layout,
        )
        self.attn = ChunkedGeometricProductAttention(algebra, config, layout)
        self.ffn_norm = CliffordLayerNorm(
            algebra,
            config.channels,
            eps=config.layer_norm_eps,
            recover=config.layer_norm_recover,
            layout=layout,
        )
        self.ffn = MultiRotorFFN(algebra, config, layout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.attn_norm(x), key_padding_mask=key_padding_mask))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class CliffordTinyStoriesLM(nn.Module):
    """TinyStories SLM using clifra attention and even-layout FFN blocks."""

    def __init__(self, config: SLMConfig, algebra: AlgebraContext | None = None):
        super().__init__()
        self.config = config
        self.algebra = build_algebra(config) if algebra is None else algebra
        self.layout = self.algebra.layout(config.even_grades)
        self.lane_dim = self.layout.dim

        self.token_embedding = nn.Embedding(config.vocab_size, config.channels * self.lane_dim)
        self.position_embedding = nn.Parameter(torch.zeros(config.seq_len, config.channels, self.lane_dim))
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(SLMBlock(self.algebra, config, self.layout) for _ in range(config.num_layers))
        self.final_norm = CliffordLayerNorm(
            self.algebra,
            config.channels,
            eps=config.layer_norm_eps,
            recover=config.layer_norm_recover,
            layout=self.layout,
        )
        self.readout = nn.Linear(config.channels * self.lane_dim, config.vocab_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.01)
        nn.init.normal_(self.readout.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.readout.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> SLMOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, L], got shape {tuple(input_ids.shape)}")
        B, L = input_ids.shape
        if L > self.config.seq_len:
            raise ValueError(f"sequence length {L} exceeds configured seq_len={self.config.seq_len}")

        x = self.token_embedding(input_ids).view(B, L, self.config.channels, self.lane_dim)
        x = x + self.position_embedding[:L].unsqueeze(0)
        x = self.embedding_dropout(x)
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(
                    lambda values, current_block=block: current_block(values, key_padding_mask=key_padding_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        logits = self.readout(x.reshape(B, L, self.config.channels * self.lane_dim))

        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(f"targets must match input_ids shape {tuple(input_ids.shape)}, got {tuple(targets.shape)}")
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return SLMOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample tokens."""
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
        for _ in range(int(max_new_tokens)):
            context = input_ids[:, -self.config.seq_len :]
            logits = self(context).logits[:, -1, :]
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / float(temperature)
                if top_k is not None and int(top_k) > 0:
                    values, _ = torch.topk(logits, k=min(int(top_k), logits.shape[-1]))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
