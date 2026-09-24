"""Measure the app's first page load: time and memory, in a fresh Python process.

The Streamlit script runs inside Streamlit's AppTest harness (no browser), so the
numbers cover exactly the app's own code, including importing its libraries.

Usage, from the repo root:

    python benchmarks/bench_startup.py app.py
    python benchmarks/bench_startup.py legacy/app.py

Prints one JSON line. If the page fails to load, the error is recorded instead.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import psutil
from streamlit.testing.v1 import AppTest


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("app", help="path to the Streamlit script to measure")
    args = parser.parse_args()
    app = Path(args.app).resolve()

    # The app imports its own package from its folder; run from a temporary folder,
    # so nothing is written into the repo.
    sys.path.insert(0, str(app.parent))
    os.chdir(tempfile.mkdtemp(prefix="gd-bench-"))

    result = {"app": str(app), "rss_mb_before": rss_mb()}
    at = AppTest.from_file(str(app), default_timeout=1800)
    start = time.perf_counter()
    at.run()
    result["first_load_s"] = round(time.perf_counter() - start, 2)
    result["rss_mb_after"] = rss_mb()
    result["error"] = str(at.exception[0].value) if at.exception else None
    memory = psutil.Process().memory_info()
    if hasattr(memory, "peak_wset"):  # Windows only
        result["peak_mb"] = round(memory.peak_wset / 2**20)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
