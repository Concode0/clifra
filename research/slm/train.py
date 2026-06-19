# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Train the clifra TinyStories SLM."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clifra.optimizers import make_riemannian_optimizer

from .config import SLMConfig
from .data import build_datasets, load_corpus
from .model import CliffordTinyStoriesLM, estimate_activation_hotspots


def _positive_int_or_none(value: str) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="roneneldan/TinyStories")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--valid-split", default="validation")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/slm"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--attention-query-chunk-size", type=_positive_int_or_none, default=16)
    parser.add_argument("--ffn-versor-chunk-channels", type=_positive_int_or_none, default=32)
    parser.add_argument("--ffn-reflection-pairs", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


@torch.no_grad()
def evaluate(model: CliffordTinyStoriesLM, loader: DataLoader, *, max_batches: int = 20) -> tuple[float, float]:
    model.eval()
    losses = []
    for batch_idx, (input_ids, targets) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        output = model(input_ids, targets)
        losses.append(float(output.loss.detach().cpu()))
    mean_loss = float(sum(losses) / max(1, len(losses)))
    return mean_loss, math.exp(mean_loss) if mean_loss < 20 else float("inf")


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    config = SLMConfig(
        device=args.device,
        dtype=args.dtype,
        channels=args.channels,
        num_layers=args.layers,
        num_heads=args.heads,
        attention_query_chunk_size=args.attention_query_chunk_size,
        ffn_versor_chunk_channels=args.ffn_versor_chunk_channels,
        ffn_reflection_pairs=args.ffn_reflection_pairs,
        gradient_checkpointing=args.gradient_checkpointing,
        dropout=args.dropout,
    )
    print(f"hotspot_estimate_mib={estimate_activation_hotspots(config, batch_size=args.batch_size).as_dict()}")
    corpus = load_corpus(
        dataset_name=args.dataset_name,
        train_split=args.train_split,
        valid_split=args.valid_split,
        text_column=args.text_column,
        max_train_examples=args.max_train_examples,
        max_valid_examples=args.max_valid_examples,
    )
    if corpus.tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"byte tokenizer has vocab_size={corpus.tokenizer.vocab_size}, but config expects {config.vocab_size}"
        )
    train_ds, valid_ds = build_datasets(corpus, seq_len=config.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CliffordTinyStoriesLM(config).to(config.resolved_device, dtype=config.torch_dtype)
    optimizer = make_riemannian_optimizer(model, model.algebra, optimizer="adam", lr=args.lr)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            input_ids, targets = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            input_ids, targets = next(iterator)
        input_ids = input_ids.to(config.resolved_device)
        targets = targets.to(config.resolved_device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, targets)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val_loss, ppl = evaluate(model, valid_loader)
            print(f"step={step} train_loss={float(output.loss.detach().cpu()):.4f} val_loss={val_loss:.4f} ppl={ppl:.2f}")
            model.train()
            checkpoint = {
                "config": asdict(config),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, args.checkpoint_dir / "latest.pt")


if __name__ == "__main__":
    main()
