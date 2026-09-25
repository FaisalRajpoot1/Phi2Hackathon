"""Scoring one method's answer on one generated graph, against the answer key."""
from graph_detective import evaluate

TRUTH = {
    "rings": [["Ana", "Ben", "Cy"]],
    "people": ["Ana", "Ben", "Cy", "Dee", "Eve", "Finn"],
    "hard_negatives": {"spouses": [["Dee", "Eve"]], "busy_professional": ["Finn"]},
}
NO_RING = {"rings": [], "people": ["Ana", "Ben"], "hard_negatives": {}}


def test_correct_missed_and_false_flags_are_counted():
    score = evaluate.score(["Ana", "Ben", "Dee"], TRUTH)
    assert (score["correct"], score["missed"], score["false_flags"]) == (2, 1, 1)
    assert score["precision"] == 2 / 3 and score["recall"] == 2 / 3


def test_names_not_in_the_data_are_invented_and_count_as_false_flags():
    score = evaluate.score(["Ana", "Zed Moon"], TRUTH)
    assert score["invented"] == 1
    assert score["false_flags"] == 1


def test_honest_look_alikes_that_were_flagged_are_counted_by_type():
    score = evaluate.score(["Ana", "Dee", "Eve"], TRUTH)
    assert score["look_alikes_flagged"] == {"spouses": 2, "busy_professional": 0}
    assert score["look_alikes_total"] == {"spouses": 2, "busy_professional": 1}


def test_a_graph_with_no_ring_only_asks_for_silence():
    assert evaluate.score([], NO_RING)["false_alarm"] is False
    assert evaluate.score(["Ana"], NO_RING)["false_alarm"] is True
    assert evaluate.score([], NO_RING)["recall"] is None


def test_an_empty_answer_on_a_ring_graph_finds_nothing():
    score = evaluate.score([], TRUTH)
    assert (score["recall"], score["precision"], score["f1"]) == (0.0, None, 0.0)
