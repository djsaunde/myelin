from __future__ import annotations

import pytest
import torch

from spiker import (
    CharacterVocabulary,
    SpikeGPTConfig,
    SpikeLanguageModel,
    SpikingSequenceLIF,
    evaluate_language_model,
    sample_token_batch,
    split_token_sequence,
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


def test_character_vocabulary_round_trips_and_rejects_unknown_chars() -> None:
    vocab = CharacterVocabulary.from_text("banana")
    encoded = vocab.encode("nab")

    assert vocab.size == 3
    assert vocab.decode(encoded) == "nab"
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        vocab.encode("band")


def test_split_token_sequence_and_sample_batch() -> None:
    tokens = torch.arange(20)

    train_tokens, val_tokens = split_token_sequence(
        tokens,
        validation_fraction=0.25,
        min_validation_tokens=3,
    )
    inputs, targets = sample_token_batch(
        train_tokens,
        batch_size=4,
        context_length=3,
        device="cpu",
    )

    assert train_tokens.tolist() == list(range(15))
    assert val_tokens.tolist() == list(range(15, 20))
    assert inputs.shape == (4, 3)
    assert targets.shape == (4, 3)


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


def test_cached_forward_step_matches_full_sequence_logits() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=9,
        context_length=6,
        n_layer=2,
        n_embd=12,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (2, config.context_length))

    full_logits = model(input_ids)
    state = model.initial_state(batch_size=input_ids.shape[0])
    step_logits = []
    for step in range(input_ids.shape[1]):
        logits, state = model.forward_step(input_ids[:, step], state)
        step_logits.append(logits)

    assert isinstance(full_logits, torch.Tensor)
    assert torch.allclose(torch.stack(step_logits, dim=1), full_logits, atol=1e-6)


def test_cached_and_uncached_greedy_generation_match_within_context_window() -> None:
    torch.manual_seed(0)
    config = SpikeGPTConfig(
        vocab_size=7,
        context_length=8,
        n_layer=1,
        n_embd=8,
        dropout=0.0,
        lif_threshold=0.0,
    )
    model = SpikeLanguageModel(config)
    input_ids = torch.tensor([[0, 1, 2]])

    cached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=3, sampling="greedy", use_cache=False)

    assert torch.equal(cached, uncached)


def test_evaluate_language_model_reports_loss_bpc_and_restores_training() -> None:
    torch.manual_seed(0)
    model = SpikeLanguageModel(
        SpikeGPTConfig(
            vocab_size=8,
            context_length=4,
            n_layer=1,
            n_embd=8,
            dropout=0.0,
            lif_threshold=0.0,
        )
    )
    model.train()
    tokens = torch.randint(0, 8, (32,))

    metrics = evaluate_language_model(
        model,
        tokens,
        batch_size=2,
        context_length=4,
        device="cpu",
        batches=2,
    )

    assert metrics.loss > 0
    assert metrics.bits_per_character > 0
    assert metrics.perplexity > 1
    assert model.training


def test_spike_language_model_rejects_context_overflow() -> None:
    model = SpikeLanguageModel(SpikeGPTConfig(vocab_size=5, context_length=4))

    with pytest.raises(ValueError, match="exceeds context_length"):
        model(torch.zeros((1, 5), dtype=torch.long))
