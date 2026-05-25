"""Train a tiny SpikeGPT-style character language model."""

from __future__ import annotations

import argparse
import time

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

from spiker import SpikeGPTConfig, SpikeLanguageModel

DEFAULT_TEXT = (
    "spiking neural networks trade dense activations for sparse events. "
    "spiker explores fast training paths for those event driven models. "
)


def encode_text(text: str) -> tuple[torch.Tensor, dict[str, int], list[str]]:
    vocab = sorted(set(text))
    stoi = {char: index for index, char in enumerate(vocab)}
    encoded = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    return encoded, stoi, vocab


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = tokens.numel() - context_length - 1
    if max_start <= 0:
        raise ValueError("text is too short for the requested context length")
    starts = torch.randint(0, max_start, (batch_size,))
    inputs = torch.stack([tokens[start : start + context_length] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + context_length + 1] for start in starts])
    return inputs.to(device=device), targets.to(device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", default=DEFAULT_TEXT)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dense-embedding",
        action="store_true",
        help="use ordinary dense token embeddings instead of hard surrogate binary embeddings",
    )
    add_compile_policy_arg(parser)
    add_matmul_precision_arg(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    configure_matmul_precision(args.matmul_precision)
    tokens, _stoi, vocab = encode_text(args.text)
    config = SpikeGPTConfig(
        vocab_size=len(vocab),
        context_length=args.context_length,
        n_layer=args.layers,
        n_embd=args.embedding,
        dropout=args.dropout,
        lif_threshold=args.lif_threshold,
        spike_embedding=not args.dense_embedding,
    )
    compile_model = resolve_compile_policy(args.compile, args.device)
    raw_model = SpikeLanguageModel(config).to(device=args.device)
    print(
        "config="
        f"device:{args.device},compile:{compile_model},compile_policy:{args.compile},"
        f"context_length:{args.context_length},layers:{args.layers},embedding:{args.embedding},"
        f"batch:{args.batch},steps:{args.steps},lr:{args.lr},dropout:{args.dropout},"
        f"lif_threshold:{args.lif_threshold},"
        f"spike_embedding:{not args.dense_embedding},vocab_size:{len(vocab)}",
        flush=True,
    )
    print()
    print_model_summary(raw_model)
    print()

    model = compile_training_model(raw_model, compile_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step_times: list[float] = []

    print("| Step | Loss | Emb Spike Rate | Mean Block Spike Rate | Step ms |", flush=True)
    print("|---:|---:|---:|---:|---:|", flush=True)

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            tokens,
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
            print(
                f"| {step} | {float(loss.detach()):.6f} | "
                f"{rates['embedding']:.4f} | {mean_block_rate:.4f} | "
                f"{step_seconds * 1000:.3f} |",
                flush=True,
            )

    print()
    print_step_time_summary(step_times)


if __name__ == "__main__":
    main()
