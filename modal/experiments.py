"""Modal harness for GPU experiments while the local 5090 is busy.

Targets RTX-PRO-6000 (Blackwell, sm_120) — the same compute capability as our
local RTX 5090 — so step-time / throughput numbers are directly comparable. The
image replicates our exact uv environment (torch 2.13 nightly cu130 + Triton),
so the Triton WKV/LIF kernels and torch.compile behave identically to local.

Run:
  uv run --extra cloud modal run modal/experiments.py::gpu_info
  uv run --extra cloud modal run modal/experiments.py::compile_mode_ab
"""

from __future__ import annotations

import subprocess

import modal

APP_NAME = "myelin-experiments"
REMOTE = "/root/myelin"
VENV_PY = "/opt/venv/bin/python"
GPU = "RTX-PRO-6000"  # sm_120, same architecture as the local RTX 5090

# Dep layer (cached unless pyproject/uv.lock change), then code, then install project.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .env({"UV_PROJECT_ENVIRONMENT": "/opt/venv"})
    .add_local_file("pyproject.toml", f"{REMOTE}/pyproject.toml", copy=True)
    .add_local_file("uv.lock", f"{REMOTE}/uv.lock", copy=True)
    .add_local_file(".python-version", f"{REMOTE}/.python-version", copy=True)
    .add_local_file("README.md", f"{REMOTE}/README.md", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --extra cuda --no-install-project")
    .add_local_dir("src", f"{REMOTE}/src", copy=True)
    .add_local_dir("examples", f"{REMOTE}/examples", copy=True)
    .run_commands(f"cd {REMOTE} && uv sync --frozen --extra cuda")
)

app = modal.App(APP_NAME, image=image)


def _run(code: str) -> None:
    """Run a python snippet in the replicated uv venv and stream its output."""
    subprocess.run([VENV_PY, "-c", code], cwd=REMOTE, check=True)


@app.function(gpu=GPU, timeout=20 * 60)
def gpu_info() -> None:
    """Smoke test: confirm the GPU is sm_120 and the nightly stack imports."""
    _run(
        "import torch, triton\n"
        "print('torch', torch.__version__, '| triton', triton.__version__)\n"
        "print('device', torch.cuda.get_device_name())\n"
        "print('capability', torch.cuda.get_device_capability())\n"
        "print('arch_list', torch.cuda.get_arch_list())\n"
        "from myelin.wkv_triton import weighted_key_value_triton\n"
        "import torch as t\n"
        "k=t.randn(2,32,128,device='cuda'); v=t.randn(2,32,128,device='cuda')\n"
        "td=t.randn(128,device='cuda'); tf=t.randn(128,device='cuda')\n"
        "y=weighted_key_value_triton(k,v,td,tf); t.cuda.synchronize()\n"
        "print('triton WKV ok, out', tuple(y.shape))"
    )


@app.function(gpu=GPU, timeout=40 * 60)
def compile_mode_ab() -> None:
    """A/B torch.compile modes on the 12L step (the open throughput question)."""
    _run(
        "import time, torch\n"
        "from myelin.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; B,T=64,1024\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "def bench(mode):\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "    kw = {} if mode=='default' else {'mode': mode}\n"
        "    for i,blk in enumerate(m.blocks):\n"
        "        m.blocks[i]=torch.compile(blk, **kw)\n"
        "    opt=torch.optim.AdamW(m.parameters(),lr=1e-4)\n"
        "    ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "    ts=[]\n"
        "    for i in range(14):\n"
        "        torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "        opt.zero_grad(set_to_none=True)\n"
        "        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=m(inp,tgt)\n"
        "        loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "        if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "    ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "    del m,opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    return ms,peak\n"
        "print(torch.cuda.get_device_name(),'| 12L/512d ctx1024 B64 bf16 regional-compile')\n"
        "print('| mode | step ms | tok/s | peak GB |'); print('|---|--:|--:|--:|')\n"
        "for mode in ['default','reduce-overhead','max-autotune-no-cudagraphs']:\n"
        "    try:\n"
        "        ms,peak=bench(mode)\n"
        "        print(f'| {mode} | {ms:.1f} | {B*T/ms*1000:,.0f} | {peak:.2f} |',flush=True)\n"
        "    except Exception as e:\n"
        "        print(f'| {mode} | ERR {type(e).__name__}: {str(e)[:50]} | | |',flush=True)"
    )
