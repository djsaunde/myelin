"""Train a tiny SpikeGPT-style character language model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from example_utils import (
    add_compile_policy_arg,
    add_matmul_precision_arg,
    compile_training_model,
    configure_matmul_precision,
    print_model_summary,
    print_step_time_summary,
    resolve_compile_policy,
)

from spiker import (
    ByteVocabulary,
    CharacterVocabulary,
    SpikeGPTConfig,
    SpikeLanguageModel,
    evaluate_language_model,
    sample_token_batch,
    split_token_sequence,
)

DEFAULT_TEXT = (
    "spiking neural networks trade dense activations for sparse events. "
    "spiker explores fast training paths for those event driven models. "
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument(
        "--vocab",
        choices=("char", "byte"),
        default="char",
        help="tokenization mode; byte uses a fixed 256-token UTF-8 vocabulary",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-tokens", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--lif-threshold",
        type=float,
        default=0.0,
        help=(
            "LIF threshold for the tiny demo; SpikeGPTConfig defaults to 1.0, "
            "which is better suited to longer contexts and larger activations"
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--sample-prompt", default="spik")
    parser.add_argument("--sample-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dense-embedding",
        action="store_true",
        help="use ordinary dense token embeddings instead of hard surrogate binary embeddings",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="checkpoint SpikeGPT blocks during training to reduce saved activations",
    )
    add_compile_policy_arg(parser)
    add_matmul_precision_arg(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    configure_matmul_precision(args.matmul_precision)
    text = args.text_file.read_text(encoding="utf-8") if args.text_file is not None else args.text
    vocabulary = (
        CharacterVocabulary.from_text(text)
        if args.vocab == "char"
        else ByteVocabulary.from_text(text)
    )
    tokens = vocabulary.encode(text)
    train_tokens, val_tokens = split_token_sequence(
        tokens,
        validation_fraction=args.val_fraction,
        min_validation_tokens=args.min_val_tokens,
    )
    config = SpikeGPTConfig(
        vocab_size=vocabulary.size,
        context_length=args.context_length,
        n_layer=args.layers,
        n_embd=args.embedding,
        dropout=args.dropout,
        lif_threshold=args.lif_threshold,
        spike_embedding=not args.dense_embedding,
        gradient_checkpointing=args.activation_checkpointing,
    )
    compile_model = resolve_compile_policy(args.compile, args.device)
    raw_model = SpikeLanguageModel(config).to(device=args.device)
    print(
        "config="
        f"device:{args.device},compile:{compile_model},compile_policy:{args.compile},"
        f"vocab:{args.vocab},"
        f"context_length:{args.context_length},layers:{args.layers},embedding:{args.embedding},"
        f"batch:{args.batch},steps:{args.steps},lr:{args.lr},dropout:{args.dropout},"
        f"lif_threshold:{args.lif_threshold},"
        f"spike_embedding:{not args.dense_embedding},"
        f"activation_checkpointing:{args.activation_checkpointing},"
        f"vocab_size:{vocabulary.size},"
        f"train_tokens:{train_tokens.numel()},val_tokens:{val_tokens.numel()}",
        flush=True,
    )
    print()
    print_model_summary(raw_model)
    print()

    model = compile_training_model(raw_model, compile_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_times: list[float] = []

    print(
        "| Step | Train Loss | Val Loss | Val BPC | Val PPL | "
        "Emb Spike Rate | Mean Block Spike Rate | Step ms |",
        flush=True,
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_token_batch(
            train_tokens,
            batch_size=args.batch,
            context_length=args.context_length,
            device=args.device,
        )
        torch_device = torch.device(args.device)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        start = time.perf_counter()
        optimizer.zero_grad()
        loss, _logits = model(inputs, targets)
        loss.backward()
        optimizer.step()
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        step_seconds = time.perf_counter() - start
        step_times.append(step_seconds)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            rates = raw_model.spike_rates(inputs)
            block_rates = [value for key, value in rates.items() if key != "embedding"]
            mean_block_rate = sum(block_rates) / len(block_rates) if block_rates else 0.0
            eval_metrics = None
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                eval_metrics = evaluate_language_model(
                    raw_model,
                    val_tokens,
                    batch_size=args.batch,
                    context_length=args.context_length,
                    device=args.device,
                    batches=args.eval_batches,
                )
            print(
                f"| {step} | {float(loss.detach()):.6f} | "
                f"{'' if eval_metrics is None else f'{eval_metrics.loss:.6f}'} | "
                f"{'' if eval_metrics is None else f'{eval_metrics.bits_per_character:.4f}'} | "
                f"{'' if eval_metrics is None else f'{eval_metrics.perplexity:.4f}'} | "
                f"{rates['embedding']:.4f} | {mean_block_rate:.4f} | "
                f"{step_seconds * 1000:.3f} |",
                flush=True,
            )

    print()
    print_step_time_summary(step_times)
    prompt = args.sample_prompt
    try:
        prompt_token_ids = vocabulary.encode(prompt)
    except ValueError as exc:
        print(
            f"sample_skipped={exc}",
            flush=True,
        )
        return
    prompt_tokens = prompt_token_ids.unsqueeze(0).to(device=args.device)
    generated = raw_model.generate(
        prompt_tokens,
        max_new_tokens=args.sample_tokens,
        top_k=min(8, vocabulary.size),
        sampling="greedy",
    )
    print(f"sample={vocabulary.decode(generated[0].cpu())!r}", flush=True)


if __name__ == "__main__":
    main()
