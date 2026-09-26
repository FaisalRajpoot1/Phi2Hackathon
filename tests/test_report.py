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


def test_the_share_of_flags_that_were_right_is_shown():
    lines = report.table([row(correct=3, missed=1, false_flags=1)]).splitlines()
    assert "3 of 4 (75%)" in lines[2]  # ring people found, and flags that were right


def test_the_table_can_be_limited_to_some_graph_sizes():
    rows = [row(correct=2, missed=1), {**row(correct=5, missed=5), "size": 300}]
    assert "2 of 3" in report.table(rows, sizes={10}) and "5 of" not in report.table(rows, sizes={10})


def test_the_readme_tables_match_the_raw_results():
    readme = report.README.read_text(encoding="utf-8")
    for name, text in report.tables(report.load_rows()).items():
        block = f"<!-- {name}:start -->\n{text}\n<!-- {name}:end -->"
        assert block in readme, f"The README's {name} table is out of date: run python benchmarks/report.py --write"


def test_models_run_on_the_live_demos_gpu_get_their_own_line():
    rows = [row(), row(method="space:microsoft/Phi-4-mini-instruct", correct=1, missed=1)]
    lines = report.table(rows).splitlines()
    assert "Phi-4-mini-instruct on a free GPU, AI only" in lines[3]


def test_the_table_has_one_line_per_method_and_data_kind():
    rows = [row(correct=2, missed=1), row(method="gemini:gemini-x", correct=1, missed=2),
            row(method="gemini:gemini-x", mode="identity", correct=3)]
    lines = report.table(rows).splitlines()
    assert len(lines) == 2 + 3  # a header, a separator, and three lines
    assert "2 of 3" in lines[2]
