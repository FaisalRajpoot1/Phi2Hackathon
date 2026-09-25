"""Compare the detector with AI-only detection, on generated data with an answer key.

    python benchmarks/bench_detection.py --methods detector gemini:gemini-3.5-flash-lite local:microsoft/phi-2
    python benchmarks/bench_detection.py --methods detector --dev      # development seeds only
    python benchmarks/bench_detection.py --timing                      # the detector on very large graphs

Each output row is one method's answer on one graph (and one run), with its score
against the answer key. The runner can stop and start again: rows that are already
in the output are skipped. Every AI method gets the same compact data, instructions
and JSON answer format. An answer that can't be read counts as an empty answer; a
prompt too long for the model is recorded as "too long", never cut down.

Gemini calls are spaced to use at most half of the free per-minute limit, and the
run stops at the daily limit. Both come from benchmarks/gemini_limits.json.
"""
import argparse
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

# Runs from anywhere: the graph_detective package lives at the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graph_detective import data, evaluate, graph, llm, pipeline, synth  # noqa: E402

MODES = {"insurance": "claims", "identity": "identity"}
TEST_SEEDS = [(2001, 1), (2002, 1), (2003, 1), (2004, 1), (2005, 0)]  # (seed, rings): 4 with a ring, 1 without
DEV_SEEDS = [(seed, 1) for seed in range(1000, 1010)]
SIZES = {"detector": [10, 20, 40, 100, 300], "gemini": [10, 20, 40, 100, 300], "local": [10, 20, 40]}
RUNS = {"detector": 1, "gemini": 3, "local": 1}
TIMING_SIZES = [1_000, 10_000, 100_000]
LIMITS_FILE = ROOT / "benchmarks" / "gemini_limits.json"


class StopRun(Exception):
    """Something the runner can't score (a missing key, a daily limit, a network error)."""


def generate(mode, seed, size, rings):
    make = synth.insurance if mode == "insurance" else synth.identity
    tree, truth = make(seed=seed, rings=rings, **({"claims": size} if mode == "insurance" else {"holders": size}))
    return data.load(tree, MODES[mode], f"{mode}-{size}-{seed}"), truth


class Gemini:
    """Calls one Gemini model, spaced out and within the daily limit."""

    def __init__(self, model, calls_today):
        limits = json.loads(LIMITS_FILE.read_text())[model]
        self.model, self.gap = model, 60 / (limits["rpm"] / 2)
        self.left_today, self.last_call = limits["rpd"] - calls_today, 0.0

    def __call__(self, dataset):
        if self.left_today <= 0:
            raise StopRun(f"The daily limit for {self.model} is reached; run again tomorrow to continue.")
        time.sleep(max(0.0, self.last_call + self.gap - time.monotonic()))
        self.last_call, self.left_today = time.monotonic(), self.left_today - 1
        return pipeline.gemini_analyst(model=self.model)(dataset)


def analyst_for(method, rows):
    if method == "detector":
        return pipeline.detector_analyst()
    kind, _, model = method.partition(":")
    if kind == "gemini":
        today = datetime.date.today().isoformat()
        return Gemini(model, sum(1 for row in rows.values() if row["method"] == method and row["date"] == today))
    if kind == "local":
        return pipeline.local_analyst(model)
    raise SystemExit(f"Unknown method: {method}")


def answer(analyst, dataset):
    """(names, error, seconds). Only errors that are real answers are returned."""
    start = time.perf_counter()
    try:
        names, error = list(analyst(dataset)), None
    except llm.BadAnswer:
        names, error = [], "unreadable"
    except llm.PromptTooLong:
        names, error = [], "too long"
    except StopRun:
        raise
    except Exception as problem:
        raise StopRun(f"{type(problem).__name__}: {problem}") from problem
    return names, error, round(time.perf_counter() - start, 3)


def load_rows(path):
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows[(row["method"], row["mode"], row["size"], row["seed"], row["run"])] = row
    return rows


def run_quality(methods, seeds, out):
    rows = load_rows(out)
    for method in methods:
        family = method.partition(":")[0]
        analyst = analyst_for(method, rows)
        for mode in MODES:
            for size in SIZES[family]:
                for seed, rings in seeds:
                    for run in range(1, RUNS[family] + 1):
                        key = (method, mode, size, seed, run)
                        if key in rows:
                            continue
                        dataset, truth = generate(mode, seed, size, rings)
                        try:
                            names, error, seconds = answer(analyst, dataset)
                        except StopRun as stop:
                            print(f"STOPPED at {key}: {stop}")
                            return
                        row = {"method": method, "mode": mode, "size": size, "seed": seed, "run": run,
                               "date": datetime.date.today().isoformat(), "names": names, "error": error,
                               "seconds": seconds, "prompt_chars": len(pipeline.build_prompt(dataset)),
                               **evaluate.score(names, truth)}
                        rows[key] = row
                        with out.open("a", encoding="utf-8") as file:
                            file.write(json.dumps(row) + "\n")
                        print(method, mode, size, seed, run, "->", error or f"f1 {row['f1']}", f"{seconds}s")


def run_timing(out):
    for size in TIMING_SIZES:
        dataset, truth = generate("insurance", 3001, size, 3)
        seconds = []
        for _ in range(5):
            start = time.perf_counter()
            rings = graph.detect(dataset)
            seconds.append(time.perf_counter() - start)
        flagged = [person for ring in rings for person in ring.members]
        row = {"size": size, "records": len(dataset.records), "median_s": round(statistics.median(seconds), 3),
               "all_s": [round(s, 3) for s in seconds], **evaluate.score(flagged, truth)}
        with out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print("detector timing", size, "claims ->", row["median_s"], "s median")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["detector"])
    parser.add_argument("--dev", action="store_true", help="use the development seeds")
    parser.add_argument("--timing", action="store_true", help="time the detector on very large graphs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = ROOT / "benchmarks" / "results"
    if args.timing:
        run_timing(args.out or results / "round2_detector_timing.jsonl")
    else:
        default = "round2_detection_dev.jsonl" if args.dev else "round2_detection.jsonl"
        run_quality(args.methods, DEV_SEEDS if args.dev else TEST_SEEDS, args.out or results / default)


if __name__ == "__main__":
    main()
