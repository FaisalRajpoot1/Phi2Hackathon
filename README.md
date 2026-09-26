# Graph Detective (Phi2Hackathon)

[![tests](https://github.com/FaisalRajpoot1/Phi2Hackathon/actions/workflows/tests.yml/badge.svg?branch=fork-improvements)](https://github.com/FaisalRajpoot1/Phi2Hackathon/actions/workflows/tests.yml)

> **This is a fork.** Graph Detective was built in 24 hours by team **"The Phi Generation"** for lablab.ai's [Phi-2 Technology: 24 Hours Challenge](https://lablab.ai/event/phi-2-technology-24-hours-challenge/the-phi-generation/phi-generation-graph-detective) (February 2024), where it won **1st place**. The original code is by **Michael Lively** ([qaillc/Phi2Hackathon](https://github.com/qaillc/Phi2Hackathon)). I, Muhammad Faisal, was on the team. This fork makes the app run again on today's libraries, fixes the bugs that its tests prove, adds a fraud detector that it measures against AI, and runs a free live demo. See [What this fork changes](#what-this-fork-changes).
>
> **Try it live on a free GPU:** [huggingface.co/spaces/Faisal87/graph-detective](https://huggingface.co/spaces/Faisal87/graph-detective)

## What the app does

It draws insurance claims, or bank accounts, as a network: people on one side, claims or account details on the other. A fraud ring shows up as the same people turning up together again and again. Click **Search for Fraud**:
- By default, **the graph detector** finds the rings with plain rules. It needs no AI and no key.
- Or an AI looks for the ring on its own, as in the original app: Gemini (with a free API key) or Phi-2 on your own computer.

Every name is checked against the data, and the ring is drawn in red. In the [live demo](https://huggingface.co/spaces/Faisal87/graph-detective), Phi-4-mini also explains each ring, and every reason it gives is checked too.

## What this fork changes

### Round 1: make it run again, and prove the old bugs

By September 2026, the original app no longer started, for two separate reasons:

| Problem | Error today |
|---|---|
| `app.py` imports `layout.py`, which was never added to the repo | `ModuleNotFoundError: No module named 'layout'` |
| The bundled 2024 copy of CrewAI needs the 2024 LangChain, and `requirements.txt` did not pin it | `ModuleNotFoundError: No module named 'langchain.agents.format_scratchpad'` (LangChain 1.4.2) |

Google's side changed too. Support for the `google-generativeai` library ended on 30 November 2025, and the model the app asked for is gone: the original call now fails with `404 models/gemini-pro is not found`. The same data through this fork's Gemini code named exactly Persons 1 to 6, with no invented names. (That sample gives its answer away, though: the ring is Accidents 1 to 5 and the lowest person numbers. Round 2 compares methods on data that doesn't.)

**The bugs, proven with tests.** `tests/test_legacy_bugs.py` runs the unchanged original code, kept in [`legacy/`](legacy/), with fakes in place of the 5 GB model. Each bug is a strict expected failure, so it stays documented; the same check must pass on the new code.

| Bug in the original | Fix in this fork |
|---|---|
| The Phi-2 tool uses `yield`, so the agent got a generator object, never Phi-2's text | The Phi call returns the text |
| Importing the tool loaded the whole model, before anyone asked for it | The model loads on first use |
| It asked for 2,048 new tokens, which alone fills Phi-2's 2,048-token context | 256 new tokens; a prompt that doesn't fit is refused, never cut |
| It passed `temperature=0.75` without sampling, so the temperature was ignored | Greedy decoding, with no temperature |
| The Gemini tool crashed at import when no key was set | The key is read when a call is made |
| `app.py` imports `layout.py`, which is missing | Removed; a credit footer instead |
| Three agents were defined, but only one ran | All three roles run: analyst, verifier and graph-maker |

Other changes:
- **No more bundled CrewAI.** The three roles now run as a light pipeline on Google's current library, `google-genai`. Gemini must answer in JSON that matches a schema, and the pipeline waits and retries when Gemini is busy. The current CrewAI (1.15) would have added two vector databases, telemetry, and PDF and Excel readers that this app doesn't use.
- **People are keyed by name.** The original keyed graph nodes by role label ("Driver 1"), so two different people with the same label became one node.
- **Every name is checked.** Names the analyst invents are shown as a warning, and left out of the ring.
- **Local models load in bfloat16**, which halves Phi-2's memory on a CPU.

#### Measured results (round 1)

| Measure | Original | This fork | Change |
|---|---|---|---|
| Starts today? | No: two import errors | Yes, with no key | — |
| Install size (`site-packages`) | 1.67 GB, 197 packages | 1.15 GB, 80 packages | −0.52 GB (−31%) |
| First page load (median of 5) | Stops after 4.0 s with an error | 1.04 s, 95 MiB | — |
| When Phi-2 loads | At start-up, in float32 | When first used, in bfloat16 | — |
| Phi-2 peak memory on a CPU (median of 3) | 10,195 MiB | 5,721 MiB | −44% |
| Phi-2 load time on a CPU (median of 3) | 22.4 s | 13.0 s | −42% |
| Phi-2 speed on a CPU (median of 3) | 1.11 tokens/s | 1.84 tokens/s | 1.7× faster |
| Tests | None | 41 tests, plus the 7 proven bugs | — |

How to read this:
- The install is smaller because 117 of the original's 197 packages were unused (for example spaCy, FAISS, yfinance and Clarifai). My first version pinned torch 2.8.0, whose Windows CPU build alone is 1.32 GB, and that made the install *bigger* than the original's. torch 2.14.0 is 526 MB.
- float32 is slower on this laptop because at 10 GB, Windows moves part of the model to disk: after loading, only 0.4 to 4.5 GB of it stayed in memory.
- Speed depends on power and the torch version: on battery with torch 2.8.0, float32 was faster (1.51 against 1.04 tokens/s). The memory saving held in every session. The raw files explain each session.
- The first page load is fast partly because PyTorch is only imported when Phi is first used.

How it was measured: an Intel Core i7-8665U laptop (4 cores), 16 GB RAM, no GPU, Windows 11, Python 3.12, plugged in, with nothing else running and the network off. Each run is a fresh Python process; float32 and bfloat16 runs alternated. The raw data and the notes for every session are in [`benchmarks/results/`](benchmarks/results/). Scripts: `benchmarks/bench_phi_load.py`, `bench_startup.py` and `bench_install_size.py`.

### Round 2: find the rings with a detector, and measure it against AI

The original idea was to let AI agents find the ring. This round asks how well that works, compared with a plain graph rule.

What was built:
- **A graph detector** (`graph_detective/graph.py`). Two people are linked when the claims or details they share add up to 2 or more; a shared SSN or account counts double. Linked groups that meet in the same claim are one ring. Three rules drop honest look-alikes:
  - In claims, a ring needs at least 3 people who are not professionals, since a couple in two accidents is common.
  - In bank accounts, a ring needs a shared SSN or account, since families share a home and a phone.
  - A lawyer's or adjuster's links count only if they meet at least 20% of the people they meet again. An honest busy adjuster meets new people every time; a corrupt one keeps working with the same few.
- **A test-data generator with an answer key** (`graph_detective/synth.py`). It uses realistic random names in a shuffled order, plants rings, and adds honest look-alikes: busy professionals, couples in two accidents, innocent victims in staged accidents, and households that share a home and a phone. The hackathon's sample couldn't be used for a fair test, because it gives its answer away: the ring is Accidents 1 to 5 and the lowest person numbers.
- **Scoring and a benchmark runner** (`graph_detective/evaluate.py`, `benchmarks/bench_detection.py`). Every method gets the same data, the same instructions and the same answer format.

How it was measured:
- I tuned the detector's rules only on development seeds (1000 to 1009). Then I tagged the code as `round2-freeze` and ran the test seeds (2001 to 2005) once: for each kind of data and each size, 4 graphs with a ring and 1 without.
- Sizes: 10, 20, 40, 100 and 300 claims or account holders. The Phi models ran only the three small sizes: Phi-2 can read at most 2,048 tokens.
- Gemini ran 3 times on each graph. The detector and the Phi models ran once, because they give the same answer every time.
- An answer that couldn't be read counts as an empty answer. A prompt too long for the model is counted separately, not as an answer.

**Every method on the same 30 small graphs:**

<!-- round2-small:start -->
| Method | Data | Ring people found | Flags that were right | Honest look-alikes flagged | False alarms (graphs with no ring) | Invented names | Unreadable / too long | Median time per graph |
|---|---|---|---|---|---|---|---|---|
| Graph detector | Insurance claims | 21 of 36 (58%) | 21 of 23 (91%) | busy professional 0 of 50 (0%); spouses 2 of 40 (5%); victim 0 of 33 (0%) | 0 of 3 | 0 | 0 / 0 of 15 | 0.001 s |
| Graph detector | Bank accounts | 46 of 46 (100%) | 46 of 46 (100%) | household 0 of 84 (0%) | 0 of 3 | 0 | 0 / 0 of 15 | 0.000 s |
| Gemini `gemini-3.5-flash-lite`, AI only | Insurance claims | 69 of 108 (64%) | 69 of 199 (35%) | busy professional 124 of 150 (83%); spouses 6 of 120 (5%); victim 0 of 99 (0%) | 9 of 9 | 0 | 0 / 0 of 45 | 11.9 s |
| Gemini `gemini-3.5-flash-lite`, AI only | Bank accounts | 138 of 138 (100%) | 138 of 381 (36%) | household 240 of 252 (95%) | 9 of 9 | 1 | 0 / 0 of 45 | 12.0 s |
| Phi-4-mini-instruct on a free GPU, AI only | Insurance claims | 15 of 36 (42%) | 15 of 46 (33%) | busy professional 27 of 50 (54%); spouses 2 of 40 (5%); victim 0 of 33 (0%) | 3 of 3 | 0 | 1 / 0 of 15 | 2.4 s |
| Phi-4-mini-instruct on a free GPU, AI only | Bank accounts | 27 of 46 (59%) | 27 of 138 (20%) | household 40 of 84 (48%) | 3 of 3 | 1 | 2 / 0 of 15 | 3.8 s |
| phi-2 on a free GPU, AI only | Insurance claims | 10 of 24 (42%) | 10 of 51 (20%) | busy professional 16 of 30 (53%); spouses 4 of 20 (20%); victim 2 of 22 (9%) | 2 of 2 | 0 | 2 / 5 of 15 | 6.0 s |
| phi-2 on a free GPU, AI only | Bank accounts | 32 of 46 (70%) | 32 of 178 (18%) | household 45 of 84 (54%) | 2 of 3 | 0 | 3 / 0 of 15 | 4.5 s |
<!-- round2-small:end -->

**The detector and Gemini on all sizes, up to 300:**

<!-- round2-all:start -->
| Method | Data | Ring people found | Flags that were right | Honest look-alikes flagged | False alarms (graphs with no ring) | Invented names | Unreadable / too long | Median time per graph |
|---|---|---|---|---|---|---|---|---|
| Graph detector | Insurance claims | 33 of 60 (55%) | 33 of 73 (45%) | busy professional 0 of 291 (0%); spouses 40 of 240 (17%); victim 0 of 55 (0%) | 1 of 5 | 0 | 0 / 0 of 25 | 0.001 s |
| Graph detector | Bank accounts | 74 of 74 (100%) | 74 of 74 (100%) | household 0 of 575 (0%) | 0 of 5 | 0 | 0 / 0 of 25 | 0.001 s |
| Gemini `gemini-3.5-flash-lite`, AI only | Insurance claims | 102 of 180 (57%) | 102 of 685 (15%) | busy professional 563 of 873 (64%); spouses 19 of 720 (3%); victim 0 of 165 (0%) | 15 of 15 | 1 | 0 / 0 of 75 | 11.9 s |
| Gemini `gemini-3.5-flash-lite`, AI only | Bank accounts | 208 of 222 (94%) | 208 of 1786 (12%) | household 1175 of 1725 (68%) | 15 of 15 | 20 | 0 / 0 of 75 | 12.1 s |
<!-- round2-all:end -->

How to read this:
- **The detector's flags are almost always right**: 91% for claims and 100% for bank accounts on the small graphs. It raised no false alarms, never invented a name, and took about a millisecond per graph. On 100,000 claims, it takes 9 seconds.
- **AI alone finds a similar share of the ring, but flags many honest people.** On the small graphs, Gemini flagged 83% of the busy honest lawyers and adjusters and 95% of the honest households, and it raised an alarm on every graph that had no ring. Phi-4-mini and Phi-2 made the same kind of mistakes.
- **Where AI did better:** on claims, Gemini found more ring members (64% against 58%). The detector misses a ring of just 2 people plus a corrupt lawyer, because of its rule against flagging couples. By chance, all 4 insurance test rings were the smallest size (3 people), and 2 of them had that shape.
- **Why some answers were unreadable:** each one started a correct JSON list, then kept adding names until the model reached its 256-token limit.
- **A home advantage:** the test data comes from my own generator, and I tuned the detector on its development seeds. So read the detector's scores as a best case. The pattern that holds across every AI method, flagging people who merely look busy, is the finding worth keeping.
- `gemini-3.8-flash` was planned as well, but the free tier stopped after its first answer, so it is left out. The raw files explain this and every other detail.

### Round 3: a free live demo

**Live:** [huggingface.co/spaces/Faisal87/graph-detective](https://huggingface.co/spaces/Faisal87/graph-detective)

- The Gradio app (`space/app.py`) runs on Hugging Face's free ZeroGPU. The detector finds the rings on the CPU, and a Plotly graph shows them in red. **Phi-4-mini explains** each ring on the GPU. Every reason must name real people and a claim or detail they really share; anything else is left out and counted.
- The same Space served the benchmark above. Its `ai_only` API lets Phi-2 or Phi-4-mini look for rings on their own, with the same prompt as on a laptop. One Phi-2 answer took about 9 minutes on the laptop (using 2 of its 8 threads) and 4.5 to 6 seconds through the Space, including the network.
- Two problems showed up only on the live Space, and tests now catch both:
  - Hugging Face installs Gradio with extras that need pydantic 2.12.5 or older, while `google-genai` needs 2.12.5 or newer. So `space/requirements.txt` pins 2.12.5, and CI installs Gradio the same way.
  - ZeroGPU runs GPU functions in another process, and replaces an error they raise with a different one. So GPU functions now return errors as values, and a test loads the app with a decorator that behaves like ZeroGPU.

### Tests

- 104 tests pass (2 more need a flag: the tiny model, and a real Gemini call). GitHub Actions runs them on every push, in two jobs: the Streamlit app's setup, and the live Space's exact versions. They cover the new code, the Streamlit app (with Streamlit's AppTest), the Gradio app, the benchmark and the report, and a real transformers run with a tiny random Phi model of a few MB.
- The 7 bugs of the original stay documented as strict expected failures.
- Only the slow or paid parts are faked: loading the model and the Gemini client. A real Gemini test runs when you set `RUN_REAL_GEMINI=1` and a key.
- A test checks that the README's round 2 tables match the raw results; `python benchmarks/report.py --write` rebuilds them.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Setup

1. Use Python 3.12.
2. Install: `pip install -r requirements.txt`
3. Optional, for Gemini: get a free key at [aistudio.google.com](https://aistudio.google.com) and set it as the environment variable `GEMINI_API_KEY`, or paste it into the sidebar. On the free tier, Google may use prompts to improve its products, so send only made-up data, like the samples in `data/`.
4. Run: `streamlit run app.py`

The detector needs nothing else. The first Phi-2 run downloads the model (5.6 GB).

## The original hackathon description

*This is the team's description from the hackathon, unchanged.*

Leveraging revolutionary Agent and Phi-2 technology, Graph Detective uncovers concealed linkages and discerns patterns, enabling pinpoint fraud detection and prevention across insurance, banking, and eCommerce with unparalleled efficiency.

Graph Detective harnesses the power of state-of-the-art Agent and Phi-2 technology to redefine fraud detection and prevention in the insurance, banking, and eCommerce sectors. By unveiling hidden connections and identifying patterns, it offers an unrivaled solution that combines Generative AI Agents with Phi-2's speed and precision, enabling investigators to tackle fraud with unparalleled efficiency and accuracy.

This groundbreaking app not only simplifies data analysis by allowing direct interaction with the Generative AI Agent, bypassing the need for complex coding or manual analysis, but also stands out for its ability to deliver consistent results without the common pitfalls of data hallucination, requiring minimal tuning.

With fraud impacting the financial sectors to the tune of $308 billion annually in the United States alone, Graph Detective is poised to revolutionize the industry by offering an unmatched efficiency improvement over traditional methods, ensuring it remains at the forefront of meeting the evolving needs of fraud detection and prevention professionals.

What sets our app apart is its unique integration of Phi-2 and Generative AI Agents, providing a solution that not only quickly identifies fraud patterns but is also easy to implement within Agent frameworks (such as CrewAI and LangGraph), making it a game-changer in the fight against sophisticated fraud schemes.
