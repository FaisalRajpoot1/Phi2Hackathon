"""Behaviour tests for space/app.py, the Gradio app of the free GPU demo.

Only the language model is faked; the detector, Donna's check and the drawing run
for real. These tests need Gradio, so they are skipped where it is not installed.
"""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("gradio")

from graph_detective import llm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RING_1_TO_6 = {f"Person {n}" for n in range(1, 7)}


@pytest.fixture(scope="module")
def app():
    spec = importlib.util.spec_from_file_location("space_app", ROOT / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def red_labels(figure, app):
    labels = []
    for trace in figure.data:
        if trace.marker is not None and trace.marker.color == app.RING_COLOR:
            labels += list(trace.text)
    return set(labels)


def test_find_rings_on_the_insurance_sample(app):
    figure, table, dataset = app.find_rings(app.SAMPLE_NAMES[0], None, "Insurance claims")
    assert all(person in table for person in RING_1_TO_6)
    assert red_labels(figure, app) == RING_1_TO_6


def test_find_rings_in_an_uploaded_bank_file(app, tmp_path):
    upload = tmp_path / "accounts.json"
    upload.write_text((ROOT / "data" / "fraud2.json").read_text(encoding="utf-8"), encoding="utf-8")
    _, table, dataset = app.find_rings(None, str(upload), "Bank accounts")
    assert dataset.kind == "identity"
    assert all(holder in table for holder in ("Account Holder 1", "Account Holder 2", "Account Holder 3"))


def test_data_with_no_ring_says_so(app, tmp_path):
    upload = tmp_path / "claims.json"
    upload.write_text(json.dumps({"name": "Claims", "children": [
        {"name": "Claim 1", "children": [{"name": "Ana", "role": "driver"}]},
        {"name": "Claim 2", "children": [{"name": "Ben", "role": "driver"}]}]}), encoding="utf-8")
    _, table, _ = app.find_rings(None, str(upload), "Insurance claims")
    assert "No ring found" in table


def test_the_explanation_shows_only_checked_reasons(app, monkeypatch):
    answer = {"reasons": [{"people": ["Person 1", "Person 2"], "shared": "Accident 3", "why": "Repeat pair."},
                          {"people": ["Person 1", "Person 99"], "shared": "Accident 3", "why": "Made up."}]}
    monkeypatch.setattr(llm, "local_generate", lambda prompt, model_id, max_new_tokens: json.dumps(answer))
    _, _, dataset = app.find_rings(app.SAMPLE_NAMES[0], None, "Insurance claims")
    text = app.explain_rings(dataset)
    assert "Accident 3" in text and "Repeat pair." in text
    assert "Made up." not in text and "1 reason" in text


def test_the_ai_only_api_reads_answers_like_the_benchmark(app, monkeypatch):
    monkeypatch.setattr(llm, "local_generate", lambda prompt, model_id: '{"suspects": ["Person 1"]}')
    tree = (ROOT / "data" / "fraud.json").read_text(encoding="utf-8")
    answer = app.ai_only(tree, "claims", llm.PHI2)
    assert answer["names"] == ["Person 1"] and answer["error"] is None


def test_the_ai_only_api_keeps_the_raw_text_of_an_unreadable_answer(app, monkeypatch):
    monkeypatch.setattr(llm, "local_generate", lambda prompt, model_id: "Person 1 is suspicious")
    tree = (ROOT / "data" / "fraud.json").read_text(encoding="utf-8")
    answer = app.ai_only(tree, "claims", llm.PHI2)
    assert (answer["names"], answer["error"], answer["raw"]) == ([], "unreadable", "Person 1 is suspicious")


def test_the_ai_only_api_reports_a_prompt_that_does_not_fit(app, monkeypatch):
    def too_long(prompt, model_id):
        raise llm.PromptTooLong("too long for Phi-2")

    monkeypatch.setattr(llm, "local_generate", too_long)
    tree = (ROOT / "data" / "fraud.json").read_text(encoding="utf-8")
    assert app.ai_only(tree, "claims", llm.PHI2)["error"] == "too long"
