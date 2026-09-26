"""Graph Detective: the Gradio app of the free Hugging Face GPU demo.

Based on Graph Detective by Michael Lively and team "The Phi Generation": 1st place,
lablab.ai "Phi-2 Technology: 24 Hours Challenge" (February 2024). The graph detector
finds the rings on the CPU. Phi-4-mini then explains them on a free ZeroGPU GPU, and
every reason is checked against the data. For the benchmark, an API also lets Phi-2
or Phi-4-mini look for rings on their own.
"""
import json
import os
import time
from pathlib import Path

import gradio as gr
import networkx as nx
import plotly.graph_objects as go
import spaces

from graph_detective import data, explain, graph, llm, pipeline

SOURCE_URL = "https://github.com/FaisalRajpoot1/Phi2Hackathon"
RING_COLOR, PERSON_COLOR, CONTEXT_COLOR = "#E53935", "#90A4AE", "#FDD835"
SAMPLES = {
    "Insurance claims (the hackathon's sample)": "fraud.json",
    "Bank accounts (the hackathon's sample)": "fraud2.json",
}
SAMPLE_NAMES = list(SAMPLES)
KINDS = {"Insurance claims": "claims", "Bank accounts": "identity"}
MODELS = (llm.PHI2, llm.PHI4_MINI)
MAX_RECORDS = 20_000

# ZeroGPU wants the models on cuda when the app starts; a real GPU is attached only
# while a @spaces.GPU function runs. Elsewhere the models load on first use.
ON_ZERO_GPU = os.getenv("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
if ON_ZERO_GPU:
    llm.DEVICE = "cuda"
    for model_id in MODELS:
        llm.load_local_model(model_id)


def load_input(sample, upload, kind):
    if upload:
        dataset = data.load(json.loads(Path(upload).read_text(encoding="utf-8")), KINDS[kind], Path(upload).name)
    else:
        dataset = data.load_sample(SAMPLES[sample])
    if len(dataset.records) > MAX_RECORDS:
        raise gr.Error(f"That file is too big for the free demo (more than {MAX_RECORDS:,} links).")
    return dataset


def figure(dataset, ring):
    """The network: people as circles (the ring in red), claims or details as squares."""
    network = nx.Graph((("person", r.person), ("context", r.context)) for r in dataset.records)
    position = nx.spring_layout(network, seed=0)
    edge_x, edge_y = [], []
    for a, b in network.edges():
        edge_x += [position[a][0], position[b][0], None]
        edge_y += [position[a][1], position[b][1], None]
    chart = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", showlegend=False,
                                 line={"color": "#CFD8DC", "width": 1}))
    groups = [("Claims and details", [n for n in network if n[0] == "context"], CONTEXT_COLOR, "square", "markers"),
              ("Other people", [n for n in network if n[0] == "person" and n[1] not in ring], PERSON_COLOR,
               "circle", "markers"),
              ("The ring", [n for n in network if n[0] == "person" and n[1] in ring], RING_COLOR, "circle",
               "markers+text")]
    for name, nodes, color, symbol, mode in groups:
        if nodes:
            chart.add_trace(go.Scatter(x=[position[n][0] for n in nodes], y=[position[n][1] for n in nodes],
                                       mode=mode, name=name, text=[n[1] for n in nodes], hoverinfo="text",
                                       textposition="top center", marker={"color": color, "symbol": symbol, "size": 12}))
    chart.update_layout(height=560, margin={"l": 10, "r": 10, "t": 10, "b": 10}, xaxis={"visible": False},
                        yaxis={"visible": False}, legend={"orientation": "h"}, plot_bgcolor="white")
    return chart


def rings_text(rings):
    if not rings:
        return "**No ring found.** No group of people keeps turning up together often enough."
    lines = []
    for number, ring in enumerate(rings, start=1):
        lines.append(f"**Ring {number}:** {', '.join(ring.members)}")
        for (a, b), shared in sorted(ring.evidence.items(), key=lambda item: graph.natural_key(" ".join(item[0]))):
            lines.append(f"- {a} and {b} share: {', '.join(shared)}")
    return "\n".join(lines)


def find_rings(sample, upload, kind):
    dataset = load_input(sample, upload, kind)
    rings = graph.detect(dataset)
    ring_people = {person for ring in rings for person in ring.members}
    return figure(dataset, ring_people), rings_text(rings), dataset


@spaces.GPU(duration=60)
def gpu_generate(prompt, model_id, max_new_tokens=llm.MAX_NEW_TOKENS):
    """Runs on the GPU. Returns (text, error). A prompt that doesn't fit is returned,
    not raised: ZeroGPU runs this function in another process, and turns an exception
    into a different one that only keeps the class name."""
    try:
        return llm.local_generate(prompt, model_id=model_id, max_new_tokens=max_new_tokens), None
    except llm.PromptTooLong as error:
        return "", str(error)


def explain_rings(dataset):
    if dataset is None:
        raise gr.Error("Find the rings first.")
    rings = graph.detect(dataset)
    if not rings:
        return "There is no ring to explain."

    def ask(prompt, schema):
        text, error = gpu_generate(prompt, llm.PHI4_MINI, explain.EXPLAIN_TOKENS)
        if error:
            raise llm.PromptTooLong(error)
        return explain.read_reasons(text)

    result = explain.explain(dataset, rings, ask)
    if result.error:
        return f"The AI could not explain the rings: {result.error}"
    lines = ["**Why they look like a ring**, written by Phi-4-mini. Every reason was checked against the data:"]
    lines += [f"- **{', '.join(reason.people)}** ({reason.shared}): {reason.why}" for reason in result.kept]
    if not result.kept:
        lines.append("- No reason passed the check.")
    if result.rejected:
        count = len(result.rejected)
        lines.append(f"\n{count} reason{'s' if count > 1 else ''} left out: it named people, or a shared "
                     "claim or detail, that are not in the data.")
    return "\n".join(lines)


def ai_only(tree_json: str, kind: str, model_id: str) -> dict:
    """For the benchmark: a model looks for the ring on its own, exactly as
    benchmarks/bench_detection.py does on a laptop. Returns the names it gave, an
    error ("unreadable" or "too long") if any, its raw text, and the seconds taken."""
    if model_id not in MODELS:
        raise gr.Error(f"Unknown model. Use one of: {', '.join(MODELS)}")
    dataset = data.load(json.loads(tree_json), kind, "api")
    start = time.perf_counter()
    names, error = [], None
    raw, too_long = gpu_generate(pipeline.build_prompt(dataset), model_id)
    if too_long:
        error = "too long"
    else:
        try:
            names = pipeline.parse_suspects(raw)
        except llm.BadAnswer:
            error = "unreadable"
    return {"names": names, "error": error, "raw": raw, "seconds": round(time.perf_counter() - start, 3)}


HEADER = f"""# Graph Detective
Find fraud rings: the same people turning up together again and again, across insurance claims or bank accounts.

- **The detector** finds the rings with plain graph rules. It needs no AI, and it runs in milliseconds.
- **Phi-4-mini** then explains them, on a free GPU. Every reason it gives is checked against the data, and anything it made up is left out.

Built at the Phi-2 hackathon by Michael Lively and team "The Phi Generation" (1st place, February 2024); this fork by Muhammad Faisal. [Source code, tests and measurements]({SOURCE_URL})
"""

with gr.Blocks(title="Graph Detective", delete_cache=(3600, 3600)) as demo:
    gr.Markdown(HEADER)
    state = gr.State()
    with gr.Row():
        with gr.Column(scale=1):
            sample = gr.Dropdown(SAMPLE_NAMES, value=SAMPLE_NAMES[0], label="Sample data")
            upload = gr.File(label="Or upload your own JSON, in the same format as the samples",
                             file_types=[".json"], type="filepath")
            kind = gr.Radio(list(KINDS), value="Insurance claims", label="What your file holds")
            find_button = gr.Button("Find rings", variant="primary")
            explain_button = gr.Button("Explain with Phi-4-mini (free GPU)")
            gr.Markdown("Use made-up data only: this is a public demo.")
        with gr.Column(scale=2):
            plot = gr.Plot(label="The network, with the rings in red")
            rings_box = gr.Markdown()
            reasons_box = gr.Markdown()
    find_button.click(find_rings, [sample, upload, kind], [plot, rings_box, state])
    explain_button.click(explain_rings, [state], [reasons_box])
    demo.load(find_rings, [sample, upload, kind], [plot, rings_box, state])
    gr.api(ai_only, api_name="ai_only")


if __name__ == "__main__":
    demo.launch()
