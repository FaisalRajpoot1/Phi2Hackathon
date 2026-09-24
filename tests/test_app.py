"""Behaviour tests for app.py. The Streamlit script runs for real inside AppTest;
only the language model is faked."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from graph_detective import llm

APP = str(Path(__file__).resolve().parents[1] / "app.py")
SEARCH = "Search for Fraud"


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def start():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def search(at):
    next(button for button in at.button if button.label == SEARCH).click().run()
    assert not at.exception, at.exception
    return at


def test_app_starts_without_a_key():
    at = start()
    assert at.title[0].value == "Graph Detective"


def test_search_without_a_key_says_what_to_do():
    at = search(start())
    assert any("GEMINI_API_KEY" in error.value for error in at.error)


def test_search_shows_checked_names_and_warns_about_invented_ones(monkeypatch):
    monkeypatch.setattr(llm, "gemini_json",
                        lambda prompt, schema, model, api_key=None: schema(suspects=["Person 1", "Person 99"]))
    at = search(start())
    assert "Person 1" in " ".join(markdown.value for markdown in at.markdown)
    assert any("Person 99" in warning.value for warning in at.warning)
