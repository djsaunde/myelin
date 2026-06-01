"""Streamlit playground for trained SpikeGPT checkpoints.

Run:  uv run --extra app streamlit run examples/app/spikegpt_playground.py

Only curated, known-good checkpoints from model_registry.json are offered (the
runs/ directory is full of scratch checkpoints we never want to expose). Config
and parameter counts are read from the checkpoint itself; the registry only
curates which models appear and their measured BPC.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import streamlit as st
import torch

from myelin import load_spike_language_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry.json"


@st.cache_data
def load_registry() -> list[dict]:
    """Curated entries whose checkpoint file actually exists, in registry order."""
    entries = json.loads(REGISTRY_PATH.read_text())
    return [e for e in entries if (REPO_ROOT / e["path"]).exists()]


@st.cache_resource(show_spinner="Loading checkpoint…")
def load_model(rel_path: str, device: str):
    checkpoint = load_spike_language_checkpoint(REPO_ROOT / rel_path, map_location=device)
    model = checkpoint.model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, checkpoint.vocabulary, checkpoint.metadata, model.config, n_params


def surprisal_bits(model, ids: torch.Tensor, prompt_len: int) -> list[tuple[int, float]]:
    """Per-token (token_id, bits) for the generated continuation under the model."""
    with torch.no_grad():
        logits = model(ids)
    logp = torch.log_softmax(logits[0].float(), dim=-1)  # [T, V]
    out: list[tuple[int, float]] = []
    for pos in range(prompt_len, ids.shape[1]):
        tok = int(ids[0, pos])
        bits = -logp[pos - 1, tok].item() / math.log(2.0)
        out.append((tok, bits))
    return out


def byte_glyph(tok: int) -> str:
    if tok == 10:
        return "⏎\n"
    if 32 <= tok < 127:
        return html.escape(chr(tok))
    return "·"


def bits_color(bits: float, vmax: float = 8.0) -> str:
    # green (confident, low bits) -> red (surprised, high bits)
    f = max(0.0, min(1.0, bits / vmax))
    r, g = int(40 + 200 * f), int(180 - 120 * f)
    return f"background-color: rgba({r},{g},60,0.30)"


st.set_page_config(page_title="SpikeGPT playground", layout="wide")
st.title("⚡ SpikeGPT playground")

registry = load_registry()
if not registry:
    st.error(f"No known-good checkpoints found. Checked {REGISTRY_PATH} against {REPO_ROOT}/runs.")
    st.stop()

with st.sidebar:
    st.header("Model")
    labels = [f"{e['name']}  —  {e['metric']}" for e in registry]
    idx = st.selectbox("Checkpoint", range(len(registry)), format_func=lambda i: labels[i])
    entry = registry[idx]
    cuda_ok = torch.cuda.is_available()
    device = st.radio(
        "Device",
        ["cpu", "cuda"] if cuda_ok else ["cpu"],
        horizontal=True,
        help="GPU may be busy with a training run; CPU is safe for short generations.",
    )

    st.header("Sampling")
    max_new = st.slider("Max new tokens", 8, 512, 128, 8)
    sampling = st.radio("Strategy", ["multinomial", "greedy"], horizontal=True)
    temperature_disabled = sampling == "greedy"
    temperature = st.slider("Temperature", 0.1, 2.0, 0.9, 0.05, disabled=temperature_disabled)
    top_k = st.slider("Top-k", 1, 256, 40, 1, disabled=temperature_disabled)
    seed = st.number_input("Seed", value=0, step=1)
    use_cache = st.checkbox("Use recurrent cache (faster)", value=True)

model, vocab, metadata, config, n_params = load_model(entry["path"], device)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", entry["dataset"])
c2.metric("Quality", entry["metric"].split("(")[0].strip())
c3.metric("Config", f"{config.n_layer}L · {config.n_embd}d · ctx {config.context_length}")
c4.metric("Params", f"{n_params / 1e6:.1f}M")
st.caption(entry["blurb"] + f"  ·  vocab={vocab.size}  ·  type={config.model_type}")

default_prompt = "The " if entry["dataset"].startswith("enwik8") else "ROMEO:"
prompt = st.text_area("Prompt", value=default_prompt, height=90)

if st.button("Generate", type="primary"):
    if not prompt:
        st.warning("Enter a prompt.")
        st.stop()
    torch.manual_seed(int(seed))
    prompt_ids = vocab.encode(prompt).unsqueeze(0).to(device)
    with st.spinner(f"Generating {max_new} tokens on {device}…"):
        out = model.generate(
            prompt_ids,
            max_new_tokens=int(max_new),
            temperature=float(temperature),
            top_k=int(top_k),
            sampling=sampling,
            use_cache=use_cache,
        )
    plen = prompt_ids.shape[1]
    continuation = vocab.decode(out[0, plen:].cpu())

    st.subheader("Generation")
    st.markdown(
        f"<div style='font-family:monospace;white-space:pre-wrap;border:1px solid #ddd;"
        f"padding:10px;border-radius:6px'>"
        f"<span style='color:#888'>{html.escape(prompt)}</span>"
        f"<span style='font-weight:600'>{html.escape(continuation)}</span></div>",
        unsafe_allow_html=True,
    )

    # SNN-distinctive readout: spike sparsity over the generated sequence
    rates = model.spike_rates(out)
    block_rates = [v for k, v in rates.items() if k != "embedding"]
    mean_block = sum(block_rates) / len(block_rates) if block_rates else 0.0
    s1, s2 = st.columns(2)
    s1.metric("Embedding spike rate", f"{rates['embedding'] * 100:.1f}%")
    s2.metric(
        "Mean block spike rate",
        f"{mean_block * 100:.1f}%",
        help="Fraction of neurons firing — sparsity is the point of an SNN.",
    )

    # per-byte surprisal heatmap over the continuation
    st.subheader("Per-byte surprisal (bits) — green = confident, red = surprised")
    pieces = []
    for tok, bits in surprisal_bits(model, out, plen):
        pieces.append(
            f"<span title='{bits:.2f} bits' style='{bits_color(bits)};padding:1px'>"
            f"{byte_glyph(tok)}</span>"
        )
    st.markdown(
        "<div style='font-family:monospace;white-space:pre-wrap;line-height:1.8'>"
        + "".join(pieces)
        + "</div>",
        unsafe_allow_html=True,
    )
    bits_only = [b for _, b in surprisal_bits(model, out, plen)]
    if bits_only:
        st.caption(
            f"mean {sum(bits_only) / len(bits_only):.2f} bits/byte over the {len(bits_only)} "
            f"generated bytes (≈ BPC of this sample)"
        )
