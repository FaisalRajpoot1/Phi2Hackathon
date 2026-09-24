"""The language model backends: local Phi (with a faked model) and Gemini (with a faked client)."""
import os
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai._gaos.errors import GenAiError
from pydantic import BaseModel

from fakes import REPLY
from graph_detective import llm


class Suspects(BaseModel):
    suspects: list[str]


class FakeInteractions:
    """Stands in for client.interactions: gives one answer per call, in order.
    An answer that is an exception is raised instead."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(output_text=answer)


def fake_client(*answers):
    return SimpleNamespace(interactions=FakeInteractions(answers))


def interactions_error(status):
    """The error class the Interactions API raises."""
    return GenAiError(f"HTTP {status}", httpx.Response(status, text="{}"))


def api_error(status):
    """The error class the older generate_content API raises."""
    return genai_errors.ClientError(status, {"error": {"message": f"HTTP {status}"}})


# ---- Gemini ----------------------------------------------------------------

def test_gemini_returns_the_schema_object():
    client = fake_client('{"suspects": ["Person 1"]}')
    assert llm.gemini_json("Find the ring.", Suspects, client=client) == Suspects(suspects=["Person 1"])


def test_gemini_asks_for_the_schema_and_nothing_is_stored():
    client = fake_client('{"suspects": []}')
    llm.gemini_json("Find the ring.", Suspects, client=client)
    call = client.interactions.calls[0]
    assert call["model"] == llm.DEFAULT_GEMINI_MODEL
    assert call["input"] == "Find the ring."
    assert call["response_format"]["schema"] == Suspects.model_json_schema()
    assert call["store"] is False


@pytest.mark.parametrize("busy", [interactions_error(429), api_error(429)])
def test_gemini_waits_and_retries_when_busy(busy):
    client = fake_client(busy, busy, '{"suspects": []}')
    waits = []
    llm.gemini_json("Find the ring.", Suspects, client=client, sleep=waits.append)
    assert len(client.interactions.calls) == 3
    assert len(waits) == 2 and waits[0] < waits[1]


def test_gemini_gives_up_after_the_retries():
    client = fake_client(*[interactions_error(429)] * 5)
    with pytest.raises(llm.GeminiBusy):
        llm.gemini_json("Find the ring.", Suspects, client=client, retries=4, sleep=lambda seconds: None)
    assert len(client.interactions.calls) == 5


@pytest.mark.parametrize("missing", [interactions_error(404), api_error(404)])
def test_retired_model_gets_a_clear_message(missing):
    client = fake_client(missing)
    with pytest.raises(llm.ModelUnavailable, match="gemini-9-old"):
        llm.gemini_json("Find the ring.", Suspects, client=client, model="gemini-9-old")


def test_unreadable_answer_keeps_the_raw_text():
    client = fake_client("I think Person 1 is suspicious.")
    with pytest.raises(llm.BadAnswer) as caught:
        llm.gemini_json("Find the ring.", Suspects, client=client)
    assert caught.value.raw == "I think Person 1 is suspicious."


def test_gemini_needs_a_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(llm.MissingKey):
        llm.gemini_json("Find the ring.", Suspects)


def test_gemini_reads_the_key_when_called(monkeypatch):
    # The key is set after the module was imported; it must still be used.
    made = []
    monkeypatch.setattr(llm.genai, "Client",
                        lambda api_key: made.append(api_key) or fake_client('{"suspects": []}'))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    llm.gemini_json("Find the ring.", Suspects)
    assert made == ["test-key"]


# ---- Local Phi -------------------------------------------------------------

def test_phi2_gets_its_instruct_format(local_model):
    llm.local_generate("Find the ring.")
    assert local_model.tokenizer.texts[-1] == "Instruct: Find the ring.\nOutput:"


def test_a_chat_model_gets_its_chat_template(local_model):
    local_model.tokenizer.chat_template = "Phi-4-mini's template"
    assert llm.local_generate("Find the ring.", model_id=llm.PHI4_MINI) == REPLY
    assert local_model.tokenizer.texts[-1] == "<|user|>Find the ring.<|end|><|assistant|>"


def test_model_loads_once_and_without_remote_code(local_model):
    llm.local_generate("Find the ring.")
    llm.local_generate("Find it again.")
    assert [name for name, _ in local_model.loads] == [llm.PHI2, llm.PHI2]  # one tokenizer, one model
    assert not any(kwargs.get("trust_remote_code") for _, kwargs in local_model.loads)


def test_too_long_prompt_is_refused_not_cut(local_model):
    with pytest.raises(llm.PromptTooLong):
        llm.local_generate("word " * 1800)
    assert local_model.model.calls == []


TINY_PHI = "hf-internal-testing/tiny-random-PhiForCausalLM"


@pytest.mark.skipif(not os.environ.get("RUN_TINY_MODEL"),
                    reason="downloads a tiny test model (a few MB) from Hugging Face; set RUN_TINY_MODEL=1")
def test_real_transformers_path_with_a_tiny_phi():
    # Random weights, so the text is nonsense, but every real transformers call runs.
    llm.load_local_model.cache_clear()
    try:
        assert isinstance(llm.local_generate("Find the ring.", model_id=TINY_PHI, max_new_tokens=8), str)
    finally:
        llm.load_local_model.cache_clear()
