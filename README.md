# Graph Detective (Phi2Hackathon)

[![tests](https://github.com/FaisalRajpoot1/Phi2Hackathon/actions/workflows/tests.yml/badge.svg?branch=fork-improvements)](https://github.com/FaisalRajpoot1/Phi2Hackathon/actions/workflows/tests.yml)

> **This is a fork.** Graph Detective was built in 24 hours by team **"The Phi Generation"** for lablab.ai's [Phi-2 Technology: 24 Hours Challenge](https://lablab.ai/event/phi-2-technology-24-hours-challenge/the-phi-generation/phi-generation-graph-detective) (February 2024), where it won **1st place**. The original code is by **Michael Lively** ([qaillc/Phi2Hackathon](https://github.com/qaillc/Phi2Hackathon)). I, Muhammad Faisal, was on the team. This fork makes the app run again on today's libraries, fixes the bugs that its tests prove, and measures every change. See [What this fork changes](#what-this-fork-changes).

## What the app does

It draws insurance claims, or bank accounts, as a network: people on one side, claims or account details on the other. A fraud ring shows up as the same people turning up together again and again. Click **Search for Fraud**, and an analyst names the suspects: Gemini (with a free API key) or Phi-2 on your own computer. Every name is checked against the data, and the suspects are drawn in red.

## What this fork changes

### Round 1: make it run again, and prove the old bugs

By September 2026, the original app no longer started, for two separate reasons:

| Problem | Error today |
|---|---|
| `app.py` imports `layout.py`, which was never added to the repo | `ModuleNotFoundError: No module named 'layout'` |
| The bundled 2024 copy of CrewAI needs the 2024 LangChain, and `requirements.txt` did not pin it | `ModuleNotFoundError: No module named 'langchain.agents.format_scratchpad'` (LangChain 1.4.2) |

Google's side changed too: the app asked for `gemini-pro`, which Google has retired, through the `google-generativeai` library, whose support ended on 30 November 2025.

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

### Tests

- 41 tests run on every push on GitHub Actions: the new code, the Streamlit app (with Streamlit's AppTest), and a real transformers run with a tiny random Phi model of a few MB.
- The 7 bugs of the original stay documented as strict expected failures.
- Only the slow or paid parts are faked: loading the model and the Gemini client. A real Gemini test runs when you set `RUN_REAL_GEMINI=1` and a key.

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Setup

1. Use Python 3.12.
2. Install: `pip install -r requirements.txt`
3. Optional, for Gemini: get a free key at [aistudio.google.com](https://aistudio.google.com) and set it as the environment variable `GEMINI_API_KEY`, or paste it into the sidebar. On the free tier, Google may use prompts to improve its products, so send only made-up data, like the samples in `data/`.
4. Run: `streamlit run app.py`

The first Phi-2 run downloads the model (5.6 GB).

## The original hackathon description

*This is the team's description from the hackathon, unchanged.*

Leveraging revolutionary Agent and Phi-2 technology, Graph Detective uncovers concealed linkages and discerns patterns, enabling pinpoint fraud detection and prevention across insurance, banking, and eCommerce with unparalleled efficiency.

Graph Detective harnesses the power of state-of-the-art Agent and Phi-2 technology to redefine fraud detection and prevention in the insurance, banking, and eCommerce sectors. By unveiling hidden connections and identifying patterns, it offers an unrivaled solution that combines Generative AI Agents with Phi-2's speed and precision, enabling investigators to tackle fraud with unparalleled efficiency and accuracy.

This groundbreaking app not only simplifies data analysis by allowing direct interaction with the Generative AI Agent, bypassing the need for complex coding or manual analysis, but also stands out for its ability to deliver consistent results without the common pitfalls of data hallucination, requiring minimal tuning.

With fraud impacting the financial sectors to the tune of $308 billion annually in the United States alone, Graph Detective is poised to revolutionize the industry by offering an unmatched efficiency improvement over traditional methods, ensuring it remains at the forefront of meeting the evolving needs of fraud detection and prevention professionals.

What sets our app apart is its unique integration of Phi-2 and Generative AI Agents, providing a solution that not only quickly identifies fraud patterns but is also easy to implement within Agent frameworks (such as CrewAI and LangGraph), making it a game-changer in the fight against sophisticated fraud schemes.
