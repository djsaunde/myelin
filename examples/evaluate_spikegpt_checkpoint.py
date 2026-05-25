"""Evaluate and sample from a saved SpikeGPT-style checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from spiker import (
    evaluate_language_model,
    evaluate_language_model_strided,
    load_spike_language_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-stride", type=int)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--random-eval",
        action="store_true",
        help="use random validation windows instead of deterministic strided windows",
    )
    parser.add_argument("--prompt", default="spik")
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sampling", choices=("greedy", "multinomial"), default="greedy")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the recurrent cached generation path",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = load_spike_language_checkpoint(args.checkpoint, map_location=args.device)
    model = checkpoint.model.to(device=args.device)
    vocabulary = checkpoint.vocabulary
    config = model.config

    print(
        "config="
        f"device:{args.device},"
        f"context_length:{config.context_length},layers:{config.n_layer},"
        f"embedding:{config.n_embd},vocab_size:{vocabulary.size},"
        f"use_cache:{args.use_cache}",
        flush=True,
    )
    if checkpoint.metadata:
        print(f"checkpoint_metadata={checkpoint.metadata}", flush=True)

    eval_text = (
        args.text_file.read_text(encoding="utf-8") if args.text_file is not None else args.text
    )
    if eval_text:
        eval_tokens = vocabulary.encode(eval_text)
        try:
            if args.random_eval:
                metrics = evaluate_language_model(
                    model,
                    eval_tokens,
                    batch_size=args.batch,
                    context_length=config.context_length,
                    device=args.device,
                    batches=args.eval_batches,
                )
                eval_mode = "random"
            else:
                metrics = evaluate_language_model_strided(
                    model,
                    eval_tokens,
                    batch_size=args.batch,
                    context_length=config.context_length,
                    device=args.device,
                    stride=args.eval_stride,
                )
                eval_mode = "strided"
        except ValueError as exc:
            print(f"eval_skipped={exc}", flush=True)
        else:
            print(f"eval_mode={eval_mode}", flush=True)
            print("| Loss | BPC | PPL |", flush=True)
            print("|---:|---:|---:|", flush=True)
            print(
                f"| {metrics.loss:.6f} | {metrics.bits_per_character:.4f} | "
                f"{metrics.perplexity:.4f} |",
                flush=True,
            )

    try:
        prompt_tokens = vocabulary.encode(args.prompt)
    except ValueError as exc:
        print(f"sample_skipped={exc}", flush=True)
        return
    generated = model.generate(
        prompt_tokens.unsqueeze(0).to(device=args.device),
        max_new_tokens=args.sample_tokens,
        temperature=args.temperature,
        top_k=min(args.top_k, vocabulary.size),
        sampling=args.sampling,
        use_cache=args.use_cache,
    )
    print(f"sample={vocabulary.decode(generated[0].cpu())!r}", flush=True)


if __name__ == "__main__":
    main()
