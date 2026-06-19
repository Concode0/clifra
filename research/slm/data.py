# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""TinyStories data utilities backed by Hugging Face datasets and byte tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


class ByteTokenizer:
    """UTF-8 byte tokenizer with a fixed 256-token vocabulary."""

    vocab_size = 256
    eos_token_id = ord("\n")

    def encode(self, text: str) -> list[int]:
        """Encode text to byte ids."""
        return list(text.encode("utf-8"))

    def decode(self, ids: Iterable[int]) -> str:
        """Decode byte ids to text, replacing invalid sequences."""
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class TinyStoriesCorpus:
    """Tokenized TinyStories split."""

    train_ids: torch.Tensor
    valid_ids: torch.Tensor
    tokenizer: ByteTokenizer


class NextTokenDataset(Dataset):
    """Sliding-window next-token dataset."""

    def __init__(self, token_ids: Sequence[int] | torch.Tensor, *, seq_len: int):
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError(f"token_ids must be 1D, got shape {tuple(ids.shape)}")
        if ids.numel() < seq_len + 1:
            raise ValueError(f"need at least seq_len + 1 tokens, got {ids.numel()} for seq_len={seq_len}")
        self.token_ids = ids.contiguous()
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(self.token_ids.numel() - self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        window = self.token_ids[index : index + self.seq_len + 1]
        return window[:-1].clone(), window[1:].clone()


def load_corpus(
    *,
    dataset_name: str = "roneneldan/TinyStories",
    train_split: str = "train",
    valid_split: str = "validation",
    text_column: str = "text",
    tokenizer: ByteTokenizer | None = None,
    max_train_examples: int | None = None,
    max_valid_examples: int | None = None,
) -> TinyStoriesCorpus:
    """Load TinyStories through Hugging Face datasets and tokenize as UTF-8 bytes."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("TinyStories loading requires the slm dependency group: uv sync --group slm") from exc

    tokenizer = ByteTokenizer() if tokenizer is None else tokenizer
    train = load_dataset(dataset_name, split=train_split)
    valid = load_dataset(dataset_name, split=valid_split)
    train_ids = _tokenize_split(
        train,
        text_column=text_column,
        tokenizer=tokenizer,
        max_examples=max_train_examples,
        split_name=train_split,
    )
    valid_ids = _tokenize_split(
        valid,
        text_column=text_column,
        tokenizer=tokenizer,
        max_examples=max_valid_examples,
        split_name=valid_split,
    )
    return TinyStoriesCorpus(train_ids=train_ids, valid_ids=valid_ids, tokenizer=tokenizer)


def build_datasets(corpus: TinyStoriesCorpus, *, seq_len: int) -> tuple[NextTokenDataset, NextTokenDataset]:
    """Build train/validation sliding-window datasets."""
    return (
        NextTokenDataset(corpus.train_ids, seq_len=seq_len),
        NextTokenDataset(corpus.valid_ids, seq_len=seq_len),
    )


def _tokenize_split(
    split,
    *,
    text_column: str,
    tokenizer: ByteTokenizer,
    max_examples: int | None,
    split_name: str,
) -> torch.Tensor:
    if text_column not in split.column_names:
        raise ValueError(f"split {split_name!r} has no text column {text_column!r}; columns={split.column_names}")
    limit = None if max_examples is None else int(max_examples)
    if limit is not None and limit <= 0:
        raise ValueError(f"max_examples must be positive when provided, got {max_examples}")

    token_ids: list[int] = []
    for idx, row in enumerate(split):
        if limit is not None and idx >= limit:
            break
        text = row[text_column]
        if not isinstance(text, str):
            text = str(text)
        token_ids.extend(tokenizer.encode(text))
        token_ids.append(int(tokenizer.eos_token_id))

    if not token_ids:
        raise ValueError(f"split {split_name!r} produced no tokens")
    return torch.tensor(token_ids, dtype=torch.long)
