from unittest.mock import Mock

import pytest
import torch

from nemo.collections.asr.modules.transformer.transformer import TransformerDecoderNM
from nemo.collections.asr.modules.transformer.transformer_generators import (
    BeamSearchSequenceGenerator,
    GreedySequenceGenerator,
)
from nemo.collections.asr.parts.submodules.multitask_beam_decoding import TransformerAEDBeamInfer
from nemo.collections.asr.parts.submodules.multitask_greedy_decoding import TransformerAEDGreedyInfer
from nemo.collections.asr.parts.submodules.token_classifier import TokenClassifier


@pytest.fixture()
def deterministic_rng():
    state = torch.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.set_rng_state(state)


@pytest.fixture()
def decoder_nm(deterministic_rng):
    return TransformerDecoderNM(
        vocab_size=8,
        hidden_size=2,
        num_layers=1,
        inner_size=4,
        num_attention_heads=1,
        max_sequence_length=32,
    ).eval()


@pytest.fixture()
def nnet(decoder_nm):
    ans = (
        decoder_nm.embedding,
        decoder_nm.decoder,
        TokenClassifier(hidden_size=2, num_classes=8),
    )
    ans = tuple(m.eval() for m in ans)
    return ans


@pytest.fixture()
def inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1]], dtype=torch.long),  # decoder_input_ids
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


@pytest.fixture()
def tokenizer():
    tok = Mock()
    tok.pad = 0
    tok.bos = 1
    tok.eos = 2
    return tok


def test_greedy_decoding(inputs, nnet, deterministic_rng):
    gen = GreedySequenceGenerator(*nnet)
    output = gen(*inputs)

    assert len(output) == 2
    best_path, hypotheses = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 25)

    assert hypotheses is None


def test_temperature_sampling_decoding(inputs, nnet):
    gen = GreedySequenceGenerator(*nnet, temperature=10.0, n_samples=2)
    output = gen(*inputs)

    assert len(output) == 2
    best_path, hypotheses = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape[0] == 1

    assert isinstance(hypotheses, list)
    assert len(hypotheses) == 1
    (seq0,) = hypotheses
    assert seq0.shape[0] == 2
    assert (seq0[0] != seq0[1]).any()


def test_beam_decoding_beam_scores_false(inputs, nnet):
    gen = BeamSearchSequenceGenerator(*nnet, beam_size=2)
    output = gen(*inputs, return_beam_scores=False)

    assert len(output) == 1
    (best_path,) = output

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (26,)


def test_beam_decoding_beam_scores_true(inputs, nnet):
    gen = BeamSearchSequenceGenerator(*nnet, beam_size=2)
    output = gen(*inputs, return_beam_scores=True)

    assert len(output) == 3
    beam_paths, scores, best_path = output

    assert beam_paths is not None
    assert isinstance(beam_paths, list)
    assert len(beam_paths) == 1
    (beam_paths_seq0,) = beam_paths
    assert torch.is_tensor(beam_paths_seq0)
    assert beam_paths_seq0.shape == (2, 26)

    assert scores is not None
    assert isinstance(scores, list)
    assert len(scores) == 1
    (scores_seq0,) = scores
    assert torch.is_tensor(scores_seq0)
    assert scores_seq0.shape == (2,)

    assert best_path is not None
    assert torch.is_tensor(best_path)
    assert best_path.shape == (1, 26)


@pytest.fixture()
def prompted_inputs():
    B, T, C = 1, 5, 2
    return (
        torch.tensor([[1, 0, 2, 3, 4]], dtype=torch.long),  # prompt
        torch.ones(B, T, C, dtype=torch.float),  # encoder_hidden_states
        torch.ones(B, T, dtype=torch.float),  # encoder_input_mask
    )


def test_transformer_aed_beam_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDBeamInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    *_, (untrimmed,) = gen.beam_search(*prompted_inputs, return_beam_scores=True)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining


def test_transformer_aed_greedy_infer_strips_prompt(prompted_inputs, decoder_nm, nnet, tokenizer):
    decoder_input_ids, encoder_hidden_states, encoder_input_mask = prompted_inputs
    decoder_input_ids = torch.tensor([[1, 0, 2, 3, 4]], dtype=torch.long)  # prompt
    *_, classifier = nnet

    # Run the actual top-level module used by MultiTask AED model for decoding.
    # This module is expected to trim the prompt from the beginning, and eos and pad from the end.
    gen = TransformerAEDGreedyInfer(decoder_nm, classifier, tokenizer)
    ans = gen(
        encoder_hidden_states=encoder_hidden_states,
        encoder_input_mask=encoder_input_mask,
        decoder_input_ids=decoder_input_ids,
    )
    best_path = ans[0][0].y_sequence
    assert best_path is not None
    assert torch.is_tensor(best_path)

    # Now run the underlying beam search generator that doesn't trim anything.
    (untrimmed,), _ = gen.greedy_search(*prompted_inputs)
    assert untrimmed is not None
    assert torch.is_tensor(untrimmed)

    # Check that the expected trimming has indeed been done.
    torch.testing.assert_close(
        untrimmed[decoder_input_ids.shape[1] :], best_path
    )  # stripped the prompt from the beggining
