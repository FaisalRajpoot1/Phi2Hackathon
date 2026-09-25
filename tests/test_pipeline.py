"""The three roles in order: the analyst (Sam), the verifier (Donna) and the graph-maker (Henry).

In the original app, three agents were defined but only Sam ever ran
(see test_legacy_every_defined_agent_runs). Here every role must run.
"""
import pytest

from graph_detective import data, llm, pipeline


def test_every_role_runs():
    claims = data.load_sample("fraud.json")
    result = pipeline.run(claims, analyst=lambda dataset: ["Person 1", "Person 99"])
    assert result.suspects == ["Person 1", "Person 99"]  # Sam
    assert result.check.invented == ["Person 99"]         # Donna
    assert result.ring == {"Person 1"}                    # Henry draws only checked names


def test_an_analyst_error_is_reported_not_raised():
    def failing_analyst(dataset):
        raise RuntimeError("Gemini's free limit was reached.")

    result = pipeline.run(data.load_sample("fraud.json"), analyst=failing_analyst)
    assert result.error == "Gemini's free limit was reached."
    assert result.suspects == [] and result.ring == set()


def test_the_detector_finds_the_sample_ring_with_no_key():
    result = pipeline.run(data.load_sample("fraud.json"), pipeline.detector_analyst())
    assert result.ring == {f"Person {n}" for n in range(1, 7)}
    assert result.check.invented == []


def test_gemini_analyst_sends_the_compact_data(monkeypatch):
    sent = []

    def fake_gemini(prompt, schema, model, api_key=None):
        sent.append((prompt, model))
        return schema(suspects=["Person 4"])

    monkeypatch.setattr(llm, "gemini_json", fake_gemini)
    claims = data.load_sample("fraud.json")
    assert pipeline.gemini_analyst(model="gemini-test")(claims) == ["Person 4"]
    prompt, model = sent[0]
    assert data.compact_text(claims) in prompt
    assert model == "gemini-test"


def test_local_analyst_finds_the_json_in_free_text(monkeypatch):
    monkeypatch.setattr(llm, "local_generate",
                        lambda prompt, model_id: 'Sure. {"suspects": ["Person 1", "Person 2"]} Hope this helps.')
    assert pipeline.local_analyst(llm.PHI2)(data.load_sample("fraud.json")) == ["Person 1", "Person 2"]


def test_local_analyst_without_json_is_a_bad_answer(monkeypatch):
    monkeypatch.setattr(llm, "local_generate", lambda prompt, model_id: "Person 1 looks suspicious.")
    with pytest.raises(llm.BadAnswer):
        pipeline.local_analyst(llm.PHI2)(data.load_sample("fraud.json"))
