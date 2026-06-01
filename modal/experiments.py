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


@app.function(gpu=GPU, timeout=30 * 60)
def lif_bf16_bench() -> None:
    """Validate + benchmark the prototype bf16-I/O LIF vs the fp32-I/O kernel."""
    _run(
        "import time, torch\n"
        "from myelin.language import SpikingSequenceLIF\n"
        "from myelin.neurons import LIFParams, LIFState\n"
        "from myelin.triton.lif_bf16 import surrogate_lif_bf16io\n"
        "dev='cuda'; B,T,C=64,1024,512\n"
        "# --- parity: feed bf16-rounded inputs to both; isolates the bf16-storage effect\n"
        "torch.manual_seed(0)\n"
        "xb=torch.randn(B,T,C,device=dev,dtype=torch.bfloat16)\n"
        "xf=xb.float().clone().requires_grad_(True)\n"
        "ref=SpikingSequenceLIF(tau=2.0,threshold=1.0,surrogate_slope=2.0,fused=False).to(dev)\n"
        "sp_ref=ref(xf); gy=torch.randn_like(sp_ref); sp_ref.backward(gy)\n"
        "xp=xb.clone().requires_grad_(True)\n"
        "cur=xp.movedim(1,0).contiguous()\n"
        "init=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=torch.bfloat16))\n"
        "sp_p=surrogate_lif_bf16io(cur,init,LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0)).movedim(0,1)\n"
        "sp_p.backward(gy.bfloat16())\n"
        "spike_match=(sp_p.float()==sp_ref).float().mean().item()\n"
        "gref=xf.grad; gp=xp.grad.float()\n"
        "rel=((gp-gref).abs().max()/(gref.abs().max()+1e-9)).item()\n"
        "print(f'parity: spikes match {spike_match*100:.2f}%  grad rel-err {rel:.3e}')\n"
        "# --- bench fwd+bwd: bf16-I/O prototype vs fp32-I/O production kernel\n"
        "def bench(make, x):\n"
        "    for _ in range(5): _do(make,x)\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(30): _do(make,x)\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/30*1e3\n"
        "def _do(make,x):\n"
        "    xi=x.clone().requires_grad_(True); s=make(xi); s.sum().backward()\n"
        "fp32=SpikingSequenceLIF(tau=2.0,threshold=1.0,surrogate_slope=2.0,fused=True).to(dev)\n"
        "x32=torch.randn(B,T,C,device=dev)\n"
        "def proto(xi):\n"
        "    c=xi.movedim(1,0).contiguous()\n"
        "    i=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=xi.dtype))\n"
        "    return surrogate_lif_bf16io(c,i,LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0))\n"
        "ms32=bench(lambda xi: fp32(xi), x32)\n"
        "msbf=bench(proto, xb)\n"
        "print(f'LIF fwd+bwd  fp32-I/O {ms32:.3f} ms  |  "
        "bf16-I/O {msbf:.3f} ms  ({ms32/msbf:.2f}x)')"
    )


@app.function(gpu=GPU, timeout=50 * 60)
def ctx3072_sweep() -> None:
    """Batch sweep at ctx 3072 (12L/512d) — find tok/s + peak GB for the ctx-3072 repro.

    Peak GB also tells us the local 32GB ceiling (RTX-PRO-6000 here is 96GB).
    """
    _run(
        "import time, torch\n"
        "from myelin.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; T=3072\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "print(f'{torch.cuda.get_device_name()} | 12L/512d ctx{T} bf16 regional-compile')\n"
        "print('| batch | step ms | tok/s | peak GB | fits 32GB? |')\n"
        "print('|--:|--:|--:|--:|--|')\n"
        "for B in [64,72,80,88]:\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    try:\n"
        "        m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "        for i,blk in enumerate(m.blocks): m.blocks[i]=torch.compile(blk)\n"
        "        opt=torch.optim.AdamW(m.parameters(),lr=1e-4)\n"
        "        ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "        ts=[]\n"
        "        for i in range(10):\n"
        "            torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "            opt.zero_grad(set_to_none=True)\n"
        "            with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=m(inp,tgt)\n"
        "            loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "            if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "        ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "        fits='yes' if peak<31 else 'NO'\n"
        "        print(f'| {B} | {ms:.0f} | {B * T / ms * 1000:,.0f} | "
        "{peak:.1f} | {fits} |', flush=True)\n"
        "        del m,opt\n"
        "    except torch.cuda.OutOfMemoryError:\n"
        "        print(f'| {B} | OOM | | >96 | NO |',flush=True); break\n"
    )


@app.function(gpu=GPU, timeout=90 * 60)
def ctx3072_lr_probe() -> None:
    """Light LR probe for the ctx-3072 / batch-72 repro: fixed-LR arms, ~1800 steps.

    Reuses the production trainer (--lr == --lr-final → constant LR). Brackets the
    tuned 2e-3 (batch-64/ctx-1024) against 4e-3, since batch-72/ctx-3072 averages
    ~3.4x more tokens/step (less gradient noise → likely tolerates a higher LR).
    """
    import io
    import os
    import urllib.request
    import zipfile

    if not os.path.exists("/tmp/enwik8"):
        data = urllib.request.urlopen("http://mattmahoney.net/dc/enwik8.zip", timeout=180).read()
        with open("/tmp/enwik8", "wb") as f:
            f.write(zipfile.ZipFile(io.BytesIO(data)).read("enwik8")[:60_000_000])
    for lr in ("2e-3", "4e-3"):
        print(f"=== ctx3072 batch72 fixed LR {lr} ===", flush=True)
        subprocess.run(
            [
                VENV_PY,
                f"{REMOTE}/examples/train_tiny_spikegpt.py",
                "--device",
                "cuda",
                "--text-file",
                "/tmp/enwik8",
                "--vocab",
                "byte",
                "--val-fraction",
                "0.05",
                "--context-length",
                "3072",
                "--layers",
                "12",
                "--embedding",
                "512",
                "--batch",
                "72",
                "--steps",
                "1800",
                "--lr",
                lr,
                "--lr-final",
                lr,
                "--warmup-steps",
                "200",
                "--weight-decay",
                "0.1",
                "--dropout",
                "0.03",
                "--compile",
                "regional",
                "--compile-mode",
                "default",
                "--compile-warmup",
                "--log-every",
                "100",
                "--eval-every",
                "600",
            ],
            cwd=REMOTE,
            check=True,
        )


@app.function(gpu=GPU, timeout=40 * 60)
def compile_scope_ab() -> None:
    """A/B regional (per-block) vs full-model torch.compile on the 12L step."""
    _run(
        "import time, torch\n"
        "from myelin.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "torch.set_float32_matmul_precision('high')\n"
        "dev='cuda'; B,T=64,1024\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=T,n_layer=12,n_embd=512,dropout=0.03)\n"
        "def bench(scope):\n"
        "    torch.manual_seed(0); torch._dynamo.reset()\n"
        "    m=SpikeLanguageModel(cfg).to(dev).train()\n"
        "    if scope=='regional':\n"
        "        for i,blk in enumerate(m.blocks): m.blocks[i]=torch.compile(blk)\n"
        "        run=m\n"
        "    else:\n"
        "        run=torch.compile(m)\n"
        "    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True)\n"
        "    ids=torch.randint(0,256,(B,T+1),device=dev); inp,tgt=ids[:,:T],ids[:,1:]\n"
        "    ts=[]\n"
        "    for i in range(14):\n"
        "        torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "        opt.zero_grad(set_to_none=True)\n"
        "        with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=run(inp,tgt)\n"
        "        loss.backward(); opt.step(); torch.cuda.synchronize()\n"
        "        if i>=4: ts.append((time.perf_counter()-t0)*1e3)\n"
        "    ts.sort(); ms=ts[len(ts)//2]; peak=torch.cuda.max_memory_allocated()/1e9\n"
        "    del m,opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()\n"
        "    return ms,peak,float(loss)\n"
        "print('12L/512d ctx1024 B64 bf16 default-mode:')\n"
        "print('| scope | step ms | peak GB | loss |'); print('|---|--:|--:|--:|')\n"
        "for scope in ['regional','full']:\n"
        "    try:\n"
        "        ms,peak,l=bench(scope)\n"
        "        print(f'| {scope} | {ms:.1f} | {peak:.2f} | {l:.3f} |',flush=True)\n"
        "    except Exception as e:\n"
        "        print(f'| {scope} | ERR {type(e).__name__}: {str(e)[:60]} | | |',flush=True)\n"
    )


@app.function(gpu=GPU, timeout=20 * 60)
def kernel_autotune() -> None:
    """Cheap occupancy A/B: WKV BLOCK and LIF block_size at the real shapes (fwd+bwd)."""
    _run(
        "import time, torch\n"
        "import myelin.wkv_triton as W\n"
        "from myelin.triton.lif_bf16 import surrogate_lif_bf16io\n"
        "from myelin.neurons import LIFParams, LIFState\n"
        "dev='cuda'; B,T,C=64,1024,512\n"
        "def timed(fn,n=40):\n"
        "    for _ in range(8): fn()\n"
        "    torch.cuda.synchronize(); t0=time.perf_counter()\n"
        "    for _ in range(n): fn()\n"
        "    torch.cuda.synchronize(); return (time.perf_counter()-t0)/n*1e3\n"
        "k=torch.randn(B,T,C,device=dev); v=torch.randn(B,T,C,device=dev)\n"
        "td=torch.randn(C,device=dev); tf=torch.randn(C,device=dev)\n"
        "def wkv():\n"
        "    a=[x.clone().requires_grad_(True) for x in (k,v,td,tf)]\n"
        "    W.weighted_key_value_triton(*a).sum().backward()\n"
        "print('WKV fwd+bwd (B64 T1024 C512):')\n"
        "for blk in [32,64,128]:\n"
        "    W._BLOCK=blk\n"
        "    print(f'  BLOCK={blk}: {timed(wkv):.3f} ms', flush=True)\n"
        "W._BLOCK=64\n"
        "cur=torch.randn(T,B,C,device=dev,dtype=torch.bfloat16)\n"
        "par=LIFParams(tau_mem=2.0,threshold=1.0,reset=0.0)\n"
        "def lif(bs):\n"
        "    def f():\n"
        "        xi=cur.clone().requires_grad_(True)\n"
        "        init=LIFState(membrane=torch.zeros(B,C,device=dev,dtype=torch.bfloat16))\n"
        "        surrogate_lif_bf16io(xi,init,par,block_size=bs).sum().backward()\n"
        "    return f\n"
        "print('LIF bf16-I/O fwd+bwd (T1024 B64 C512):')\n"
        "for bs in [128,256,512]:\n"
        "    print(f'  block_size={bs}: {timed(lif(bs)):.3f} ms', flush=True)\n"
    )


@app.function(gpu=GPU, timeout=20 * 60)
def integrated_bf16_smoke() -> None:
    """Confirm the integrated model takes the bf16-I/O LIF path end-to-end."""
    _run(
        "import torch\n"
        "from unittest.mock import patch\n"
        "import myelin.triton.lif_bf16 as m\n"
        "from myelin.language import SpikeGPTConfig, SpikeLanguageModel\n"
        "cfg=SpikeGPTConfig(vocab_size=256,context_length=256,n_layer=2,n_embd=128,dropout=0.0)\n"
        "model=SpikeLanguageModel(cfg).to('cuda').train()\n"
        "ids=torch.randint(0,256,(4,257),device='cuda')\n"
        "calls={'n':0}; orig=m.surrogate_lif_bf16io\n"
        "def spy(*a,**k): calls['n']+=1; return orig(*a,**k)\n"
        "with patch.object(m,'surrogate_lif_bf16io',spy), "
        "patch('myelin.language.surrogate_lif_bf16io',spy):\n"
        "    with torch.autocast('cuda',dtype=torch.bfloat16):\n"
        "        loss,_=model(ids[:,:256],ids[:,1:])\n"
        "    loss.backward()\n"
        "g=sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)\n"
        'print(f\'loss={float(loss):.3f}  grad_sum={g:.1f}  bf16-LIF calls={calls["n"]} '
        "(expect {cfg.n_layer*2})')"
    )


@app.function(gpu=GPU, timeout=60 * 60)
def lif_convergence_ab() -> None:
    """Train SpikeGPT twice (fp32 LIF vs bf16-I/O LIF) and compare val curves."""
    import io
    import os
    import urllib.request
    import zipfile

    if not os.path.exists("/tmp/enwik8"):
        data = urllib.request.urlopen("http://mattmahoney.net/dc/enwik8.zip", timeout=180).read()
        raw = zipfile.ZipFile(io.BytesIO(data)).read("enwik8")[:40_000_000]
        with open("/tmp/enwik8", "wb") as f:
            f.write(raw)
    for lif in ("fp32", "bf16"):
        subprocess.run(
            [
                VENV_PY,
                f"{REMOTE}/examples/lif_convergence_ab.py",
                "--text-file",
                "/tmp/enwik8",
                "--lif",
                lif,
                "--steps",
                "4000",
            ],
            cwd=REMOTE,
            check=True,
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
