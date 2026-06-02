"""Tokenize a corpus to a uint16 ``.bin`` memmap for large-scale pretraining.

Two sources:
  * local text files (``--text-file`` repeatable) — e.g. WikiText train/valid/test;
  * a streaming HuggingFace dataset (``--hf-dataset``) — e.g. OpenWebText2.

Tokens are written with the same vocabulary the model will train under (default
the GPT-NeoX BPE), so the ``.bin`` ids match the checkpoint's tokenizer exactly.
Output is a flat ``uint16`` ``.bin`` + a ``.json`` sidecar (read by
``myelin.token_corpus.MemmapTokenCorpus``).

Examples:
  # WikiText-103 splits -> three bins
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --text-file wikitext-103/wiki.train.tokens --output data/wikitext103_train.bin
  # OpenWebText2 (streaming), capped for a validation slice
  uv run --extra tokenization python examples/prepare_token_corpus.py \\
    --hf-dataset Skylion007/openwebtext --hf-split train --text-column text \\
    --max-tokens 1_000_000_000 --output data/owt_1b.bin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from myelin import BPEVocabulary, ByteVocabulary, TokenCorpusWriter


def build_vocab(args: argparse.Namespace):
    if args.vocab == "byte":
        return ByteVocabulary()
    return BPEVocabulary.from_pretrained(args.bpe_tokenizer)


def iter_texts(args: argparse.Namespace):
    """Yield text chunks from local files or a streaming HF dataset."""
    if args.text_file:
        for path in args.text_file:
            yield Path(path).read_text(encoding="utf-8")
        return
    from datasets import load_dataset

    dataset = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True)
    for example in dataset:
        text = example.get(args.text_column)
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="output .bin path")
    parser.add_argument("--vocab", choices=("bpe", "byte"), default="bpe")
    parser.add_argument("--bpe-tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--text-file", action="append", help="local text file(s); repeatable")
    parser.add_argument("--hf-dataset", help="streaming HuggingFace dataset name")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--eos-id",
        type=int,
        default=0,
        help="token id inserted between documents (GPT-NeoX <|endoftext|>=0); -1 to disable",
    )
    parser.add_argument("--max-tokens", type=int, help="stop after this many tokens")
    parser.add_argument("--log-every", type=int, default=1000, help="log every N documents")
    args = parser.parse_args()

    if not args.text_file and not args.hf_dataset:
        parser.error("provide --text-file or --hf-dataset")

    vocab = build_vocab(args)
    start = time.perf_counter()
    docs = 0
    with TokenCorpusWriter(args.output, vocab_size=vocab.size) as writer:
        for text in iter_texts(args):
            ids = vocab.encode(text).tolist()
            if args.eos_id >= 0:
                ids.append(args.eos_id)
            writer.write(ids)
            docs += 1
            if docs % args.log_every == 0:
                rate = writer.n_tokens / max(time.perf_counter() - start, 1e-9)
                print(
                    f"docs={docs:,} tokens={writer.n_tokens:,} ({rate / 1e6:.2f}M tok/s)",
                    flush=True,
                )
            if args.max_tokens is not None and writer.n_tokens >= args.max_tokens:
                break
        total = writer.n_tokens
    print(
        f"wrote {args.output} — {total:,} tokens, vocab_size={vocab.size}, "
        f"{docs:,} docs, {time.perf_counter() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
