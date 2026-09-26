---
title: Graph Detective
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
models:
- microsoft/phi-2
- microsoft/Phi-4-mini-instruct
short_description: Find fraud rings with graph rules; Phi-4-mini explains them
---

# Graph Detective

Find fraud rings in insurance claims or bank accounts: the same people turning up together again and again.

- **The detector** finds the rings with plain graph rules. It needs no AI, and it takes milliseconds.
- **Phi-4-mini** explains them, on a free ZeroGPU GPU. Every reason is checked against the data, and anything the model made up is left out.

Graph Detective was built in 24 hours by Michael Lively and team "The Phi Generation" for lablab.ai's Phi-2 Technology: 24 Hours Challenge (February 2024), where it won 1st place. This version is by Muhammad Faisal. Source code, tests and measurements: [GitHub](https://github.com/FaisalRajpoot1/Phi2Hackathon).

This is a public demo: use made-up data only.
