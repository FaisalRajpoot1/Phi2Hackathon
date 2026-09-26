"""Build the README's round 2 tables from the raw benchmark rows.

    python benchmarks/report.py            # print the tables
    python benchmarks/report.py --write    # also update them in README.md

Every row in benchmarks/results/round2_detection*.jsonl (but not the development
file) is one method's answer on one generated graph. The tables add them up, so
every number in them can be traced back to the raw rows. The first table compares
every method on the same small graphs; the second, the detector and Gemini on all sizes.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
README = ROOT / "README.md"
FAMILIES = ["detector", "gemini", "space", "local"]
SMALL_SIZES = {10, 20, 40}
ALL_SIZES_METHODS = {"detector", "gemini:gemini-3.5-flash-lite"}
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
    lines = defaultdict(lambda: {"found": 0, "ring_people": 0, "flags": 0, "false_flags": 0, "invented": 0,
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
        line["flags"] += row["correct"] + row["false_flags"]
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
            "space": f"{model.split('/')[-1]} on a free GPU, AI only",
            "local": f"{model.split('/')[-1]} on the laptop, AI only"}[family]


def duration(seconds):
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds:.3f} s"
    return f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60:.1f} min"


def share(part, whole):
    return f"{part} of {whole} ({round(100 * part / whole)}%)" if whole else f"{part} of 0"


def table(rows, sizes=None):
    """One line per method and kind of data; `sizes` keeps only graphs of those sizes."""
    if sizes is not None:
        rows = [row for row in rows if row["size"] in sizes]
    order = lambda item: (FAMILIES.index(item[0][0].partition(":")[0]), item[0][0], list(MODES).index(item[0][1]))
    out = ["| Method | Data | Ring people found | Flags that were right | Honest look-alikes flagged "
           "| False alarms (graphs with no ring) | Invented names | Unreadable / too long | Median time per graph |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (method, mode), line in sorted(summarize(rows).items(), key=order):
        look_alikes = "; ".join(f"{kind.replace('_', ' ')} {share(flagged, total)}"
                                for kind, (flagged, total) in sorted(line["look_alikes"].items()))
        out.append(f"| {method_label(method)} | {MODES[mode]} | {share(line['found'], line['ring_people'])} "
                   f"| {share(line['found'], line['flags'])} | {look_alikes} "
                   f"| {line['false_alarms']} of {line['no_ring_graphs']} | {line['invented']} "
                   f"| {line['unreadable']} / {line['too_long']} of {line['answers']} | {duration(line['median_s'])} |")
    return "\n".join(out)


def tables(rows):
    """The README's two tables, by name."""
    return {"round2-small": table(rows, sizes=SMALL_SIZES),
            "round2-all": table([row for row in rows if row["method"] in ALL_SIZES_METHODS])}


def write_readme(blocks):
    readme = README.read_text(encoding="utf-8")
    for name, text in blocks.items():
        start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
        before, _, rest = readme.partition(start)
        _, _, after = rest.partition(end)
        readme = f"{before}{start}\n{text}\n{end}{after}"
    README.write_text(readme, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the tables in README.md")
    args = parser.parse_args()
    blocks = tables(load_rows())
    for name, text in blocks.items():
        print(f"{name}:\n{text}\n")
    if args.write:
        write_readme(blocks)


if __name__ == "__main__":
    main()
