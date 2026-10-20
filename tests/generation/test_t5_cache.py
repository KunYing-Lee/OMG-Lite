from types import SimpleNamespace, ModuleType
import sys

import pytest
import torch
from torch import nn

from omg.generation.conditions.t5 import FrozenT5TextEncoder


class Tokenizer:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def __call__(self, captions, max_length, **kwargs):
        ids = torch.zeros(len(captions), max_length, dtype=torch.long)
        for index, text in enumerate(captions):
            tokens = [ord(char) % 32 + 1 for char in text] + [1]
            n = min(len(tokens), max_length)
            ids[index, :n] = torch.tensor(tokens[:n])
        return {"input_ids": ids, "attention_mask": ids != 0}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=8)
        self.embed = nn.Embedding(34, 8)
        self.dropout = nn.Dropout(.5)
        self.calls = 0

    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.dropout(self.embed(input_ids)))


@pytest.fixture
def encoder(monkeypatch):
    transformers = ModuleType("transformers")
    transformers.T5Tokenizer = Tokenizer
    transformers.T5EncoderModel = Encoder
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return FrozenT5TextEncoder(model_name="fake-t5", max_length=6, output_dim=4,
                               frozen_eval=True, cache_max_entries=2)


def test_cache_matches_uncached_with_duplicates_mask_null_and_projection_gradients(encoder):
    encoder.train()
    assert not encoder.encoder.training and encoder.proj.training
    for captions, mask, null in [(["walk", "run", "walk"], [True, False, True], False),
                                  (["turn left", "", "jump"], [True, True, True], False),
                                  (["walk", "run"], [False, True], True)]:
        encoder.cache_max_entries = 0
        expected = encoder(captions, torch.tensor(mask), force_null_text=null)
        expected["context"].sum().backward()
        expected_grad = encoder.proj.weight.grad.clone()
        encoder.zero_grad()
        encoder.cache_max_entries = 2
        actual = encoder(captions, torch.tensor(mask), force_null_text=null)
        torch.testing.assert_close(actual["context"], expected["context"], rtol=0, atol=0)
        assert actual["mask"].equal(expected["mask"])
        actual["context"].sum().backward()
        torch.testing.assert_close(encoder.proj.weight.grad, expected_grad)
        encoder.zero_grad()
        assert len(encoder._text_cache) <= 2
    assert all(p.grad is None for p in encoder.encoder.parameters())


def test_hot_cache_skips_encoder_and_mask_is_not_cached(encoder):
    encoder(["walk", "walk"], torch.tensor([True, False]))
    calls = encoder.encoder.calls
    out = encoder(["walk", "walk"], torch.tensor([False, True]))
    assert encoder.encoder.calls == calls
    assert not out["context"][0].any() and out["context"][1].any()
    assert len(encoder._text_cache) == 1
    assert encoder.cache_misses == 1 and encoder.cache_hits == 3


def test_cache_is_transient_and_invalidated_on_weights_or_dtype_change(encoder):
    encoder(["walk"])
    state = encoder.state_dict()
    assert all("cache" not in key for key in state)
    encoder.load_state_dict(state)
    assert not encoder._text_cache
    encoder(["walk"])
    encoder.double()
    assert not encoder._text_cache
    assert encoder(["walk"])["context"].dtype == torch.float64


def test_stochastic_encoder_cannot_be_cached(encoder):
    encoder.encoder.train()
    with pytest.raises(RuntimeError, match="training mode"):
        encoder(["walk"])
    with pytest.raises(ValueError, match="frozen_eval"):
        FrozenT5TextEncoder(cache_max_entries=1)


def test_uncached_legacy_mode_and_precision_cache_separation(encoder):
    encoder.frozen_eval = False
    encoder.cache_max_entries = 0
    encoder.train()
    assert encoder.encoder.training
    encoder.frozen_eval = True
    encoder.cache_max_entries = 2
    encoder.train()
    encoder(["walk"])
    calls = encoder.encoder.calls
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoder(["walk"])
    assert encoder.encoder.calls == calls + 1
    encoder(["walk"])
    assert encoder.encoder.calls == calls + 2
