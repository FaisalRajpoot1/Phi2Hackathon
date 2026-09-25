"""The report turns raw benchmark rows into the README's tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import report  # noqa: E402


def row(method="detector", mode="insurance", correct=0, missed=0, false_flags=0, invented=0,
        false_alarm=None, error=None, seconds=1.0, flagged=None, total=None):
    return {"method": method, "mode": mode, "size": 10, "seed": 1, "run": 1, "correct": correct,
            "missed": missed, "false_flags": false_flags, "invented": invented, "false_alarm": false_alarm,
            "error": error, "seconds": seconds,
            "look_alikes_flagged": flagged or {"spouses": 0}, "look_alikes_total": total or {"spouses": 2}}


def test_counts_add_up_over_graphs():
    summary = report.summarize([row(correct=3, missed=1, false_flags=2, flagged={"spouses": 1}),
                                row(correct=2, missed=0, false_flags=0, seconds=3.0)])
    line = summary[("detector", "insurance")]
    assert (line["found"], line["ring_people"], line["false_flags"]) == (5, 6, 2)
    assert line["look_alikes"] == {"spouses": [1, 4]}
    assert line["median_s"] == 2.0


def test_a_graph_with_no_ring_counts_only_for_false_alarms():
    line = report.summarize([row(false_alarm=True), row(false_alarm=False)])[("detector", "insurance")]
    assert (line["false_alarms"], line["no_ring_graphs"]) == (1, 2)


def test_a_prompt_that_did_not_fit_is_counted_apart():
    line = report.summarize([row(missed=3, error="too long"), row(correct=3)])[("detector", "insurance")]
    assert line["too_long"] == 1
    assert (line["found"], line["ring_people"]) == (3, 3)


def test_an_unreadable_answer_counts_as_an_empty_answer():
    line = report.summarize([row(missed=3, error="unreadable")])[("detector", "insurance")]
    assert line["unreadable"] == 1
    assert (line["found"], line["ring_people"]) == (0, 3)


def test_the_table_has_one_line_per_method_and_data_kind():
    rows = [row(correct=2, missed=1), row(method="gemini:gemini-x", correct=1, missed=2),
            row(method="gemini:gemini-x", mode="identity", correct=3)]
    lines = report.table(rows).splitlines()
    assert len(lines) == 2 + 3  # a header, a separator, and three lines
    assert "2 of 3" in lines[2]
