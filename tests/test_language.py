from __future__ import annotations

import pytest
import torch

from spiker import (
    SpikeGPTConfig,
    SpikeLanguageModel,
    SpikingSequenceLIF,
    weighted_key_value,
)


def test_weighted_key_value_returns_expected_first_value_and_backpropagates() -> None:
    key = torch.randn((2, 4, 3), requires_grad=True)
    value = torch.randn((2, 4, 3), requires_grad=True)
    time_decay = torch.zeros(3, requires_grad=True)
    time_first = torch.zeros(3, requires_grad=True)

    output = weighted_key_value(key, value, time_decay, time_first)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == key.shape
    assert torch.allclose(output[:, 0], value[:, 0])
    assert key.grad is not None
    assert value.grad is not None
    assert time_decay.grad is not None
    assert time_first.grad is not None


def test_weighted_key_value_rejects_bad_shapes() -> None:
    key = torch.randn((2, 4, 3))
    value = torch.randn((2, 4, 2))

    with pytest.raises(ValueError, match="same shape"):
        weighted_key_value(key, value, torch.zeros(3), torch.zeros(3))


def test_spiking_sequence_lif_is_binary_in_hard_forward_and_has_gradients() -> None:
    inputs = torch.full((2, 3, 4), 1.2, requires_grad=True)
    lif = SpikingSequenceLIF(tau=2.0, threshold=1.0, surrogate_slope=2.0)

    spikes = lif(inputs)
    spikes.sum().backward()

    assert spikes.shape == inputs.shape
    assert set(spikes.detach().flatten().tolist()) <= {0.0, 1.0}
    assert inputs.grad is not None


def test_spike_language_model_returns_logits_loss_and_spike_rates() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=11,
        context_length=8,
        n_layer=2,
        n_embd=16,
        dropout=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.randint(0, config.vocab_size, (3, config.context_length))
    targets = torch.randint(0, config.vocab_size, (3, config.context_length))

    logits = model(input_ids)
    loss, logits_with_loss = model(input_ids, targets)
    loss.backward()
    rates = model.spike_rates(input_ids)

    assert logits.shape == (3, config.context_length, config.vocab_size)
    assert logits_with_loss.shape == logits.shape
    assert loss.ndim == 0
    assert model.embedding.weight.grad is not None
    assert "embedding" in rates
    assert "blocks.0.time" in rates
    assert "blocks.0.channel" in rates


def test_spike_language_model_generate_extends_context_and_restores_training() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7,
        context_length=4,
        n_layer=1,
        n_embd=8,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    model.train()
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])

    generated = model.generate(input_ids, max_new_tokens=3, sampling="greedy")

    assert generated.shape == (1, 9)
    assert torch.equal(generated[:, : input_ids.shape[1]], input_ids)
    assert model.training


def test_spike_language_model_rejects_context_overflow() -> None:
    model = SpikeLanguageModel(SpikeGPTConfig(vocab_size=5, context_length=4))

    with pytest.raises(ValueError, match="exceeds context_length"):
        model(torch.zeros((1, 5), dtype=torch.long))
