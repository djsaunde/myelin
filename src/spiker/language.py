"""SpikeGPT-style language-model modules.

The modules here intentionally start as ordinary PyTorch code. SpikeGPT's most
important architectural idea is to use the token sequence as the spiking time
axis and replace quadratic self-attention with an RWKV-style recurrent mixer.
Keeping the first version torch-native gives us a correctness target before we
specialize the WKV recurrence or spike operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from spiker.surrogates import SurrogateFn, atan_surrogate, hard_surrogate_spike

SpikeGPTModelType = Literal["rwkv", "rwkv-ffn-pre"]
SamplingMode = Literal["multinomial", "greedy"]


@dataclass(frozen=True)
class SpikeGPTConfig:
    """Configuration for a compact SpikeGPT-style recurrent language model."""

    vocab_size: int
    context_length: int
    n_layer: int = 4
    n_embd: int = 128
    dropout: float = 0.03
    model_type: SpikeGPTModelType = "rwkv"
    lif_tau: float = 2.0
    lif_threshold: float = 1.0
    lif_reset: float = 0.0
    surrogate_slope: float = 2.0
    spike_embedding: bool = True

    def validate(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")
        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lif_tau <= 0.0:
            raise ValueError("lif_tau must be positive")


def _time_shift(inputs: torch.Tensor) -> torch.Tensor:
    return F.pad(inputs[:, :-1], (0, 0, 1, 0))


def _mix_with_previous(
    inputs: torch.Tensor,
    mix: torch.Tensor,
) -> torch.Tensor:
    previous = _time_shift(inputs)
    return inputs * mix + previous * (1.0 - mix)


def weighted_key_value(
    key: torch.Tensor,
    value: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
) -> torch.Tensor:
    """RWKV weighted key-value recurrence.

    ``key`` and ``value`` are batch-major tensors with shape ``[B, T, C]``.
    This reference implementation matches the scalar recurrence used by the
    SpikeGPT/RWKV CUDA kernel while remaining readable and autograd-friendly.
    """

    if key.shape != value.shape:
        msg = f"key and value must have the same shape; got {key.shape} and {value.shape}"
        raise ValueError(msg)
    if key.ndim != 3:
        msg = f"key and value must have shape [B, T, C]; got {key.shape}"
        raise ValueError(msg)
    if time_decay.shape != (key.shape[-1],):
        msg = f"time_decay must have shape [{key.shape[-1]}]; got {time_decay.shape}"
        raise ValueError(msg)
    if time_first.shape != (key.shape[-1],):
        msg = f"time_first must have shape [{key.shape[-1]}]; got {time_first.shape}"
        raise ValueError(msg)

    batch, timesteps, channels = key.shape
    dtype = key.dtype
    device = key.device
    decay = -torch.exp(time_decay.to(device=device, dtype=dtype))
    first = time_first.to(device=device, dtype=dtype)
    numerator = torch.zeros((batch, channels), dtype=dtype, device=device)
    denominator = torch.zeros((batch, channels), dtype=dtype, device=device)
    log_scale = torch.full(
        (batch, channels),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )
    outputs: list[torch.Tensor] = []

    for step in range(timesteps):
        key_t = key[:, step]
        value_t = value[:, step]

        next_log_scale = torch.maximum(log_scale, first + key_t)
        old_scale = torch.exp(log_scale - next_log_scale)
        new_scale = torch.exp(first + key_t - next_log_scale)
        outputs.append(
            (old_scale * numerator + new_scale * value_t) / (old_scale * denominator + new_scale)
        )

        decayed_log_scale = log_scale + decay
        next_log_scale = torch.maximum(decayed_log_scale, key_t)
        old_scale = torch.exp(decayed_log_scale - next_log_scale)
        new_scale = torch.exp(key_t - next_log_scale)
        numerator = old_scale * numerator + new_scale * value_t
        denominator = old_scale * denominator + new_scale
        log_scale = next_log_scale

    return torch.stack(outputs, dim=1)


class SpikingSequenceLIF(nn.Module):
    """LIF unroll over a batch-major token sequence ``[B, T, C]``."""

    def __init__(
        self,
        *,
        tau: float = 2.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 2.0,
        hard_forward: bool = True,
    ) -> None:
        super().__init__()
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.threshold = threshold
        self.reset = reset
        self.surrogate = surrogate
        self.surrogate_slope = surrogate_slope
        self.hard_forward = hard_forward

    @property
    def decay(self) -> float:
        return 1.0 - (1.0 / self.tau)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            msg = f"inputs must have shape [B, T, C]; got {inputs.shape}"
            raise ValueError(msg)

        membrane = torch.zeros_like(inputs[:, 0])
        spikes: list[torch.Tensor] = []
        for step in range(inputs.shape[1]):
            membrane = membrane * self.decay + inputs[:, step]
            centered = self.surrogate_slope * (membrane - self.threshold)
            if self.hard_forward:
                spike = hard_surrogate_spike(centered, self.surrogate)
            else:
                spike = self.surrogate(centered)
            membrane = membrane * (1.0 - spike) + self.reset * spike
            spikes.append(spike)
        return torch.stack(spikes, dim=1)


class SpikeTimeMix(nn.Module):
    """RWKV time-mixing block followed by a spiking residual activation."""

    def __init__(
        self,
        n_embd: int,
        n_layer: int,
        layer_id: int,
    ) -> None:
        super().__init__()
        if n_embd <= 0:
            raise ValueError("n_embd must be positive")
        attn_size = n_embd
        layer_denominator = max(1, n_layer - 1)
        ratio_0_to_1 = layer_id / layer_denominator
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)

        position = torch.arange(n_embd, dtype=torch.float32) / n_embd
        decay_speed = -5.0 + 8.0 * torch.pow(
            torch.arange(attn_size, dtype=torch.float32) / max(1, attn_size - 1),
            0.7 + 1.3 * ratio_0_to_1,
        )
        zigzag = torch.tensor([(index + 1) % 3 - 1 for index in range(attn_size)])
        self.time_decay = nn.Parameter(decay_speed)
        self.time_first = nn.Parameter(torch.ones(attn_size) * -1.2039728 + zigzag * 0.5)
        self.time_mix_k = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_v = nn.Parameter(
            (torch.pow(position, ratio_1_to_almost0) + 0.3 * ratio_0_to_1).view(1, 1, -1)
        )
        self.time_mix_r = nn.Parameter(torch.pow(position, 0.5 * ratio_1_to_almost0).view(1, 1, -1))

        self.key = nn.Linear(n_embd, attn_size, bias=False)
        self.value = nn.Linear(n_embd, attn_size, bias=False)
        self.receptance = nn.Linear(n_embd, attn_size, bias=False)
        self.output = nn.Linear(attn_size, n_embd, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key_input = _mix_with_previous(inputs, self.time_mix_k)
        value_input = _mix_with_previous(inputs, self.time_mix_v)
        receptance_input = _mix_with_previous(inputs, self.time_mix_r)

        key = self.key(key_input)
        value = self.value(value_input)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        mixed = receptance * weighted_key_value(key, value, self.time_decay, self.time_first)
        return self.output(mixed)


class SpikeChannelMix(nn.Module):
    """RWKV channel-mixing feed-forward block."""

    def __init__(self, n_embd: int, n_layer: int, layer_id: int) -> None:
        super().__init__()
        ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)
        position = torch.arange(n_embd, dtype=torch.float32) / n_embd
        hidden_size = 4 * n_embd
        self.time_mix_k = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.time_mix_r = nn.Parameter(torch.pow(position, ratio_1_to_almost0).view(1, 1, -1))
        self.key = nn.Linear(n_embd, hidden_size, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_size, n_embd, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        key_input = _mix_with_previous(inputs, self.time_mix_k)
        receptance_input = _mix_with_previous(inputs, self.time_mix_r)
        key = torch.relu(self.key(key_input)).square()
        value = self.value(key)
        receptance = torch.sigmoid(self.receptance(receptance_input))
        return receptance * value


class SpikeGPTBlock(nn.Module):
    """SpikeGPT-style block with RWKV time/channel mixing and LIF activations."""

    def __init__(self, config: SpikeGPTConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.n_embd)
        else:
            self.ln0 = None
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        if layer_id == 0 and config.model_type == "rwkv-ffn-pre":
            self.ffn_pre = SpikeChannelMix(config.n_embd, config.n_layer, 0)
            self.att = None
        else:
            self.ffn_pre = None
            self.att = SpikeTimeMix(config.n_embd, config.n_layer, layer_id)
        self.ffn = SpikeChannelMix(config.n_embd, config.n_layer, layer_id)
        self.lif1 = SpikingSequenceLIF(
            tau=config.lif_tau,
            threshold=config.lif_threshold,
            reset=config.lif_reset,
            surrogate_slope=config.surrogate_slope,
        )
        self.lif2 = SpikingSequenceLIF(
            tau=config.lif_tau,
            threshold=config.lif_threshold,
            reset=config.lif_reset,
            surrogate_slope=config.surrogate_slope,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.ln0(inputs) if self.ln0 is not None else inputs
        if self.ffn_pre is not None:
            residual = residual + self.lif1(self.ffn_pre(self.ln1(residual)))
        else:
            if self.att is None:
                raise RuntimeError("SpikeGPTBlock has no time-mix module")
            residual = residual + self.lif1(self.att(self.ln1(residual)))
        residual = residual + self.lif2(self.ffn(self.ln2(residual)))
        return self.dropout(residual)


class SpikeLanguageModel(nn.Module):
    """SpikeGPT-like autoregressive language model."""

    def __init__(self, config: SpikeGPTConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList(
            [SpikeGPTBlock(config, layer_id) for layer_id in range(config.n_layer)]
        )
        self.ln_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=1.0e-3)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        if not self.config.spike_embedding:
            return embeddings
        return hard_surrogate_spike(
            self.config.surrogate_slope * embeddings,
            atan_surrogate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2:
            msg = f"input_ids must have shape [B, T]; got {input_ids.shape}"
            raise ValueError(msg)
        if input_ids.shape[1] > self.config.context_length:
            msg = (
                "input sequence length exceeds context_length; "
                f"got {input_ids.shape[1]} and {self.config.context_length}"
            )
            raise ValueError(msg)

        hidden = self.embed_tokens(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.head(self.ln_out(hidden))

        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        sampling: SamplingMode = "multinomial",
    ) -> torch.Tensor:
        """Autoregressively extend ``input_ids``.

        This simple reference path recomputes the context window on each token.
        A state-cached RNN path is the intended future optimization once the
        training architecture is stable.
        """

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if sampling not in ("multinomial", "greedy"):
            raise ValueError("sampling must be 'multinomial' or 'greedy'")

        output = input_ids
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = output[:, -self.config.context_length :]
            logits = self(context)
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("generate expected logits-only forward output")
            next_logits = logits[:, -1] / temperature
            if top_k is not None:
                limit = min(top_k, next_logits.shape[-1])
                values, _indices = torch.topk(next_logits, limit, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, -torch.inf),
                    next_logits,
                )
            if sampling == "greedy":
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            output = torch.cat((output, next_token), dim=1)
        if was_training:
            self.train()
        return output

    @torch.no_grad()
    def spike_rates(self, input_ids: torch.Tensor) -> dict[str, float]:
        """Return coarse spike-rate diagnostics for embeddings and block activations."""

        rates: dict[str, float] = {}
        hidden = self.embed_tokens(input_ids)
        rates["embedding"] = float(hidden.mean())
        for index, raw_block in enumerate(self.blocks):
            block = cast(SpikeGPTBlock, raw_block)
            att_input = block.ln1(block.ln0(hidden) if block.ln0 is not None else hidden)
            if block.ffn_pre is not None:
                att_spikes = block.lif1(block.ffn_pre(att_input))
            else:
                if block.att is None:
                    raise RuntimeError("SpikeGPTBlock has no time-mix module")
                att_spikes = block.lif1(block.att(att_input))
            ffn_spikes = block.lif2(block.ffn(block.ln2(hidden + att_spikes)))
            rates[f"blocks.{index}.time"] = float(att_spikes.mean())
            rates[f"blocks.{index}.channel"] = float(ffn_spikes.mean())
            hidden = block(hidden)
        return rates
