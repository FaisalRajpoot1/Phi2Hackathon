from types import SimpleNamespace

import pytest

from fakes import FakeCausalLM, FakeTokenizer


@pytest.fixture(autouse=True)
def run_in_temp_dir(monkeypatch, tmp_path):
    """Run every test in its own empty folder, so no test can write into the repo."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def local_model(monkeypatch):
    """Fakes loading a local model from Hugging Face, and records every load."""
    import transformers

    from graph_detective import llm

    fake = SimpleNamespace(tokenizer=FakeTokenizer(), model=FakeCausalLM(), loads=[])

    def counted(result):
        def from_pretrained(name, *args, **kwargs):
            fake.loads.append((name, kwargs))
            return result
        return from_pretrained

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", counted(fake.tokenizer))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", counted(fake.model))
    llm.load_local_model.cache_clear()
    yield fake
    llm.load_local_model.cache_clear()
