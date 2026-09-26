# Raw measurement data

These files are the raw output behind every number in the main README. They were measured on 24 and 25 September 2026. Local folder paths are shortened to `<temp>` (a temporary folder) and `<repo>` (this repository); no measured value was changed.

**Laptop:** Intel Core i7-8665U (4 cores, 8 threads), 16 GB RAM, no GPU, Windows 11, Python 3.12.
**Units:** memory is in MiB (`psutil`); `peak_mb` is Windows' peak working set, the most memory the process ever held.

## Round 1: make it run again

| File | What it holds | Script |
|---|---|---|
| `round1_original_imports.txt` | How the original app's imports fail today, in an environment built from `legacy/requirements.txt` | `check_original_imports.py` |
| `round1_first_page_load_original.txt` | The original app's first page load: it stops at `No module named 'layout'` | `bench_startup.py` |
| `round1_original_gemini.txt` | The original app's Gemini call (the old SDK and `gemini-pro`), made on 25 September 2026: `404 ... not found` | `check_original_gemini.py` |
| `round1_first_page_load_new.jsonl` | Five first page loads of the new app, each in a fresh Python process | `bench_startup.py` |
| `round1_install_sizes.jsonl` | Size of `site-packages`: the original's requirements (today's versions), and the new `requirements.txt` | `bench_install_size.py` |
| `round1_packages_original.txt`, `round1_packages_new.txt` | The installed packages of both environments (`pip freeze`) | — |
| `round1_phi2_load.jsonl` | **Used in the README.** Phi-2 on the CPU with torch 2.14.0: 3 runs per dtype, alternating, plugged in, nothing else running, network off | `bench_phi_load.py` |
| `round1_phi2_load_clean_battery.jsonl` | The same test on battery power, with torch 2.8.0 | `bench_phi_load.py` |
| `round1_phi2_load_first_try.jsonl`, `.txt` | The first try, with torch 2.8.0, while other tests ran at the same time | `bench_phi_load.py` |

Notes on the Phi-2 runs:
- **Memory** is stable in every session: float32 peaks at 10,190 to 10,987 MiB (about 10.7 to 11.5 GB), bfloat16 at 5,684 to 5,734 MiB (about 6.0 GB).
- **Speed depends on power and the torch version.** Plugged in, with torch 2.14.0: bfloat16 1.84 against float32 1.11 tokens/s (medians). On battery, with torch 2.8.0: bfloat16 1.04 against float32 1.51. The README uses the plugged-in runs, because they match the app's settings (torch 2.14.0).
- **One crash.** In the first try, one float32 run crashed with a segmentation fault while loading (see the `.txt` file). It did not happen again in 6 clean float32 runs, so it is not counted as a bug.
- In float32, Windows often kept only part of the model in memory after loading (`rss_mb_after_load` between 415 and 4,470 MiB, against a peak of about 10,200). Paging the rest to disk is what makes float32 slow on a 16 GB laptop.

## Round 2: the detector against AI

Every graph comes from `graph_detective/synth.py`, with its answer key. The code was frozen with the git tag `round2-freeze` before the test seeds (2001 to 2005) were run; the detector's rules were tuned only on development seeds (1000 to 1009). Each row is one method's answer on one graph, with its score (`graph_detective/evaluate.py`). `benchmarks/report.py` builds the README's tables from these files.

| File | What it holds |
|---|---|
| `round2_detection.jsonl` | The detector: 50 graphs (2 kinds of data × 5 sizes × 5 seeds), one run each |
| `round2_detection_gemini.jsonl` | Gemini `gemini-3.5-flash-lite`: the same 50 graphs, 3 runs each (150 answers), on 25 September 2026 |
| `round2_detection_space.jsonl` | Phi-2 and Phi-4-mini, on the live demo's free ZeroGPU through its `ai_only` API: 30 small graphs each (sizes 10, 20 and 40). For an unreadable answer, `raw` keeps the model's text |
| `round2_phi2_laptop_partial.jsonl` | Phi-2 on the laptop's CPU (2 of 8 threads, low priority): 8 of the 30 graphs, before Claude Code stopped the run because the laptop was low on memory. Not used in the tables |
| `round2_gemini38_stopped.txt` | The run log of `gemini-3.8-flash`: one answer, then the free tier's limit. Not used in the tables |
| `round2_detector_timing.jsonl` | The detector on 1,000, 10,000 and 100,000 claims: 5 timings each |

Notes:
- **Laptop against GPU.** 6 of Phi-2's 8 laptop answers were identical to its GPU answers. The 2 that differed were both "name almost everyone" answers: cut off on the laptop, 8 and 32 names on the GPU. Different hardware rounds the numbers slightly differently, which can change a greedy answer.
- **Unreadable answers** (Phi-2: 5, Phi-4-mini: 3). Each began a correct JSON list, then kept adding names until the model reached its 256-token limit (see `raw`).
- **Too long** (Phi-2: 5). The 40-claim insurance graphs do not fit in Phi-2's 2,048 tokens together with room for the answer.
- **Time per graph** includes the network: for Gemini, the call to Google; for the Phi models, the call to the Space.
- **The test rings.** By chance, all 4 insurance test rings are the smallest size (3 people); 2 of them are 2 people plus a corrupt lawyer, the shape the detector misses. Over 1,000 seeds, the generator draws each size from 3 to 6 equally often.
