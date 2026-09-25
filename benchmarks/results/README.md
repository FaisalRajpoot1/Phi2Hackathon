# Raw measurement data

These files are the raw output behind every number in the main README. They were measured on 24 and 25 September 2026. Local folder paths are shortened to `<temp>` (a temporary folder) and `<repo>` (this repository); no measured value was changed.

**Laptop:** Intel Core i7-8665U (4 cores, 8 threads), 16 GB RAM, no GPU, Windows 11, Python 3.12.
**Units:** memory is in MiB (`psutil`); `peak_mb` is Windows' peak working set, the most memory the process ever held.

## Round 1: make it run again

| File | What it holds | Script |
|---|---|---|
| `round1_original_imports.txt` | How the original app's imports fail today, in an environment built from `legacy/requirements.txt` | `check_original_imports.py` |
| `round1_first_page_load_original.txt` | The original app's first page load: it stops at `No module named 'layout'` | `bench_startup.py` |
| `round1_first_page_load_new.jsonl` | Five first page loads of the new app, each in a fresh Python process | `bench_startup.py` |
| `round1_install_sizes.jsonl` | Size of `site-packages`: the original's requirements (today's versions), and the new `requirements.txt` | `bench_install_size.py` |
| `round1_packages_original.txt`, `round1_packages_new.txt` | The installed packages of both environments (`pip freeze`) | — |
| `round1_phi2_load.jsonl` | **Used in the README.** Phi-2 on the CPU with torch 2.14.0: 3 runs per dtype, alternating, plugged in, nothing else running, network off | `bench_phi_load.py` |
| `round1_phi2_load_clean_battery.jsonl` | The same test on battery power, with torch 2.8.0 | `bench_phi_load.py` |
| `round1_phi2_load_first_try.jsonl`, `.txt` | The first try, with torch 2.8.0, while other tests ran at the same time | `bench_phi_load.py` |

Notes on the Phi-2 runs:
- **Memory** is stable in every session: float32 peaks at 10.2 to 11.0 GB, bfloat16 at 5.7 GB.
- **Speed depends on power and the torch version.** Plugged in, with torch 2.14.0: bfloat16 1.84 against float32 1.11 tokens/s (medians). On battery, with torch 2.8.0: bfloat16 1.04 against float32 1.51. The README uses the plugged-in runs, because they match the app's settings (torch 2.14.0).
- **One crash.** In the first try, one float32 run crashed with a segmentation fault while loading (see the `.txt` file). It did not happen again in 6 clean float32 runs, so it is not counted as a bug.
- In float32, Windows often kept only part of the model in memory after loading (`rss_mb_after_load` between 415 and 4,470 MiB, against a peak of about 10,200). Paging the rest to disk is what makes float32 slow on a 16 GB laptop.
