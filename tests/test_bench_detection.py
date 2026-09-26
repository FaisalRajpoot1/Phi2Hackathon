"""The benchmark runner: Phi models run through the live demo's GPU API, and
unreadable answers keep their raw text."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import bench_detection  # noqa: E402

from graph_detective import llm  # noqa: E402


class FakeClient:
    """Stands in for gradio_client.Client: answers /ai_only like the Space does."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def predict(self, *args, api_name):
        self.calls.append((args, api_name))
        return self.answer


def space_analyst(answer):
    analyst = bench_detection.SpaceAnalyst(llm.PHI2, client=FakeClient(answer))
    dataset, _, tree = bench_detection.generate("insurance", 2001, 10, 1)
    return analyst, dataset, tree


def test_a_space_answer_gives_the_names():
    analyst, dataset, tree = space_analyst({"names": ["Ana"], "error": None, "raw": "{...}", "seconds": 1.0})
    assert analyst(dataset) == ["Ana"]
    args, api_name = analyst.client.calls[0]
    assert api_name == "/ai_only" and args[1:] == ("claims", llm.PHI2)


def test_an_unreadable_space_answer_keeps_the_raw_text():
    analyst, dataset, _ = space_analyst({"names": [], "error": "unreadable", "raw": "Ana is odd", "seconds": 1.0})
    with pytest.raises(llm.BadAnswer) as caught:
        analyst(dataset)
    assert caught.value.raw == "Ana is odd"


def test_a_too_long_space_answer_is_a_prompt_that_did_not_fit():
    analyst, dataset, _ = space_analyst({"names": [], "error": "too long", "raw": "", "seconds": 0.1})
    with pytest.raises(llm.PromptTooLong):
        analyst(dataset)


def test_the_result_of_an_unreadable_answer_keeps_the_raw_text():
    def unreadable(dataset):
        raise llm.BadAnswer("no JSON", raw="Ana is odd")

    dataset, _, _ = bench_detection.generate("insurance", 2001, 10, 1)
    names, error, raw, _ = bench_detection.answer(unreadable, dataset)
    assert (names, error, raw) == ([], "unreadable", "Ana is odd")
