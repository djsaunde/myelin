"""Train a tiny SpikeGPT-style character language model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from example_utils import (
    add_compile_policy_arg,
    add_grad_clip_arg,
    add_matmul_precision_arg,
    add_wandb_args,
    clip_gradients,
    compile_training_model,
    configure_matmul_precision,
    finish_wandb,
    init_wandb,
    log_wandb,
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
    parser.add_argument("--weight-decay", type=float, default=0.01)
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
    parser.add_argument(
        "--compile-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run and report one compiled training step before timed logging when compile is enabled"
        ),
    )
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
    add_grad_clip_arg(parser)
    add_matmul_precision_arg(parser)
    add_wandb_args(parser)
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
        f"batch:{args.batch},steps:{args.steps},lr:{args.lr},"
        f"weight_decay:{args.weight_decay},dropout:{args.dropout},"
        f"lif_threshold:{args.lif_threshold},"
        f"grad_clip:{args.grad_clip},"
        f"compile_warmup:{args.compile_warmup},"
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config={
            "device": args.device,
            "compile": compile_model,
            "compile_policy": args.compile,
            "vocab": args.vocab,
            "context_length": args.context_length,
            "layers": args.layers,
            "embedding": args.embedding,
            "batch": args.batch,
            "steps": args.steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "lif_threshold": args.lif_threshold,
            "grad_clip": args.grad_clip,
            "compile_warmup": args.compile_warmup,
            "spike_embedding": not args.dense_embedding,
            "activation_checkpointing": args.activation_checkpointing,
            "vocab_size": vocabulary.size,
            "train_tokens": train_tokens.numel(),
            "val_tokens": val_tokens.numel(),
        },
    )
    step_times: list[float] = []

    print(
        "| Step | Train Loss | Val Loss | Val BPC | Val PPL | "
        "Emb Spike Rate | Mean Block Spike Rate | Grad Norm | Step ms |",
        flush=True,
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    try:
        model.train()
        torch_device = torch.device(args.device)
        if compile_model and args.compile_warmup:
            warmup_inputs, warmup_targets = sample_token_batch(
                train_tokens,
                batch_size=args.batch,
                context_length=args.context_length,
                device=args.device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            warmup_loss, _warmup_logits = model(warmup_inputs, warmup_targets)
            warmup_loss.backward()
            warmup_grad_norm = clip_gradients(model, args.grad_clip)
            optimizer.step()
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            warmup_seconds = time.perf_counter() - start
            print(f"compile_warmup_step_ms={warmup_seconds * 1000:.3f}", flush=True)
            print(f"compile_warmup_loss={float(warmup_loss.detach()):.6f}", flush=True)
            if warmup_grad_norm is not None:
                print(f"compile_warmup_grad_norm={warmup_grad_norm:.4f}", flush=True)
            log_wandb(
                wandb_run,
                {
                    "compile/warmup_step_ms": warmup_seconds * 1000,
                    "compile/warmup_loss": float(warmup_loss.detach()),
                    **(
                        {}
                        if warmup_grad_norm is None
                        else {"compile/warmup_grad_norm": warmup_grad_norm}
                    ),
                },
                step=0,
            )

        for step in range(1, args.steps + 1):
            inputs, targets = sample_token_batch(
                train_tokens,
                batch_size=args.batch,
                context_length=args.context_length,
                device=args.device,
            )
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, _logits = model(inputs, targets)
            loss.backward()
            grad_norm = clip_gradients(model, args.grad_clip)
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
                    f"{'' if grad_norm is None else f'{grad_norm:.4f}'} | "
                    f"{step_seconds * 1000:.3f} |",
                    flush=True,
                )
                wandb_metrics = {
                    "train/loss": float(loss.detach()),
                    "train/embedding_spike_rate": rates["embedding"],
                    "train/mean_block_spike_rate": mean_block_rate,
                    "train/step_ms": step_seconds * 1000,
                }
                if grad_norm is not None:
                    wandb_metrics["train/grad_norm"] = grad_norm
                if eval_metrics is not None:
                    wandb_metrics.update(
                        {
                            "val/loss": eval_metrics.loss,
                            "val/bpc": eval_metrics.bits_per_character,
                            "val/perplexity": eval_metrics.perplexity,
                        }
                    )
                log_wandb(wandb_run, wandb_metrics, step=step)
    finally:
        finish_wandb(wandb_run)

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
