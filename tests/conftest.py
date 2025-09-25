import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyTokenizer:
    def __call__(self, text, return_tensors=None):
        return {"text": text}


class _DummyModel:
    def __call__(self, **kwargs):
        text = kwargs.get("text", "") or ""
        normalized = text.lower()
        # Use keyword frequency as a simple stand-in for semantic similarity
        score = normalized.count("capybara")
        if score == 0:
            score = max(len(normalized), 1) / 100.0

        hidden_state = torch.tensor([[[float(score), float(score) + 0.1]]], dtype=torch.float32)
        return SimpleNamespace(last_hidden_state=hidden_state)

    def load_state_dict(self, state_dict):
        return None

    def eval(self):
        return None


@pytest.fixture
def stubbed_transformer(monkeypatch):
    """Replace the heavy RoBERTa dependencies with lightweight stubs."""

    sys.modules.pop("unified_doc_search.nlp.transformer", None)

    monkeypatch.setattr(
        "transformers.RobertaTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )

    dummy_model = _DummyModel()
    monkeypatch.setattr(
        "transformers.RobertaModel.from_pretrained",
        lambda *args, **kwargs: dummy_model,
    )
    monkeypatch.setattr("torch.load", lambda *args, **kwargs: {})

    module = importlib.import_module("unified_doc_search.nlp.transformer")
    module.model = dummy_model
    module.tokenizer = _DummyTokenizer()

    yield module

    sys.modules.pop("unified_doc_search.nlp.transformer", None)


@pytest.fixture
def flask_app(stubbed_transformer):
    sys.modules.pop("unified_doc_search.app", None)
    app_module = importlib.import_module("unified_doc_search.app")
    return app_module.app
