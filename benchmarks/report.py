"""Build the README's round 2 table from the raw benchmark rows.

    python benchmarks/report.py            # print the table
    python benchmarks/report.py --write    # also update it in README.md

Every row in benchmarks/results/round2_detection*.jsonl (but not the development
file) is one method's answer on one generated graph. The table adds them up, so
every number in it can be traced back to the raw rows.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "README.md"
START, END = "<!-- round2-table:start -->", "<!-- round2-table:end -->"
FAMILIES = ["detector", "gemini", "local"]
MODES = {"insurance": "Insurance claims", "identity": "Bank accounts"}


def load_rows():
    rows = []
    for path in sorted(RESULTS.glob("round2_detection*.jsonl")):
        if not path.stem.endswith("_dev"):
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def summarize(rows):
    """Totals for each (method, mode). A prompt that did not fit is counted apart,
    not as an answer; an unreadable answer counts as an empty answer."""
    lines = defaultdict(lambda: {"found": 0, "ring_people": 0, "false_flags": 0, "invented": 0,
                                 "look_alikes": {}, "false_alarms": 0, "no_ring_graphs": 0,
                                 "unreadable": 0, "too_long": 0, "answers": 0, "seconds": []})
    for row in rows:
        line = lines[(row["method"], row["mode"])]
        line["answers"] += 1
        if row["error"] == "too long":
            line["too_long"] += 1
            continue
        line["unreadable"] += row["error"] == "unreadable"
        line["seconds"].append(row["seconds"])
        if row["false_alarm"] is not None:
            line["no_ring_graphs"] += 1
            line["false_alarms"] += row["false_alarm"]
        line["found"] += row["correct"]
        line["ring_people"] += row["correct"] + row["missed"]
        line["false_flags"] += row["false_flags"]
        line["invented"] += row["invented"]
        for kind, flagged in row["look_alikes_flagged"].items():
            counts = line["look_alikes"].setdefault(kind, [0, 0])
            counts[0] += flagged
            counts[1] += row["look_alikes_total"][kind]
    for line in lines.values():
        seconds = line.pop("seconds")
        line["median_s"] = statistics.median(seconds) if seconds else None
    return dict(lines)


def method_label(method):
    family, _, model = method.partition(":")
    return {"detector": "Graph detector",
            "gemini": f"Gemini `{model}`, AI only",
            "local": f"{model.split('/')[-1].capitalize()} on the laptop, AI only"}[family]


def duration(seconds):
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds:.3f} s"
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def table(rows):
    order = lambda item: (FAMILIES.index(item[0][0].partition(":")[0]), item[0][0], list(MODES).index(item[0][1]))
    out = ["| Method | Data | Ring people found | False flags | Invented names | Honest look-alikes flagged "
           "| False alarms (graphs with no ring) | Unreadable / too long | Median time per graph |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (method, mode), line in sorted(summarize(rows).items(), key=order):
        look_alikes = "; ".join(f"{kind.replace('_', ' ')} {flagged} of {total}"
                                for kind, (flagged, total) in sorted(line["look_alikes"].items()))
        out.append(f"| {method_label(method)} | {MODES[mode]} | {line['found']} of {line['ring_people']} "
                   f"| {line['false_flags']} | {line['invented']} | {look_alikes} "
                   f"| {line['false_alarms']} of {line['no_ring_graphs']} "
                   f"| {line['unreadable']} / {line['too_long']} of {line['answers']} | {duration(line['median_s'])} |")
    return "\n".join(out)


def write_readme(text):
    readme = README.read_text(encoding="utf-8")
    before, _, rest = readme.partition(START)
    _, _, after = rest.partition(END)
    README.write_text(f"{before}{START}\n{text}\n{END}{after}", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the table in README.md")
    args = parser.parse_args()
    text = table(load_rows())
    print(text)
    if args.write:
        write_readme(text)


if __name__ == "__main__":
    main()
