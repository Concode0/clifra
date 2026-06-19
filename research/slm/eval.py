# clifra (C) 2026 Eunkyum Kim
# SPDX-License-Identifier: Apache-2.0

"""Evaluate or sample from a clifra TinyStories SLM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import SLMConfig
from .data import ByteTokenizer
from .model import CliffordTinyStoriesLM


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config_data = dict(checkpoint["config"])
    if args.device != "auto":
        config_data["device"] = args.device
    config_data["even_grades"] = tuple(config_data["even_grades"])
    config = SLMConfig(**config_data)

    model = CliffordTinyStoriesLM(config).to(config.resolved_device, dtype=config.torch_dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=config.resolved_device)
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output_ids[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
