"""Graph Detective: find fraud rings in insurance claims and identity data.

A fork of the Phi-2 hackathon app by Michael Lively and team "The Phi Generation"
(1st place, lablab.ai "Phi-2 Technology: 24 Hours Challenge", February 2024).

Run it with:  streamlit run app.py
"""
import streamlit as st
from streamlit_agraph import Config, agraph

from graph_detective import data, draw, llm, pipeline

DATASETS = {
    "Insurance claims": "fraud.json",
    "Insurance claims (small)": "neo4jdata.json",
    "Bank accounts": "fraud2.json",
}
ANALYSTS = {
    "Graph detector (no key needed)": "detector",
    "Gemini (needs a free key)": "gemini",
    "Phi-2 on this computer": llm.PHI2,
}

st.set_page_config(page_title="Graph Detective", layout="wide")
st.title("Graph Detective")
st.write("Find fraud rings: the same people turning up again and again, across claims or bank accounts.")

with st.sidebar:
    dataset_name = st.selectbox("Data", list(DATASETS), key="dataset")
    analyst_name = st.radio("Who looks for the ring?", list(ANALYSTS), key="analyst")
    api_key = st.text_input("Gemini API key (optional)", type="password", key="api_key",
                            help="Or set GEMINI_API_KEY. On the free tier, Google may use prompts "
                                 "to improve its products, so send only made-up data.")
    gemini_model = st.text_input("Gemini model", value=llm.DEFAULT_GEMINI_MODEL, key="gemini_model")

dataset = data.load_sample(DATASETS[dataset_name])
results = st.session_state.setdefault("results", {})

if st.button("Search for Fraud"):
    choice = ANALYSTS[analyst_name]
    if choice == "detector":
        analyst = pipeline.detector_analyst()
    elif choice == "gemini":
        analyst = pipeline.gemini_analyst(gemini_model, api_key or None)
    else:
        analyst = pipeline.local_analyst(choice)
    with st.spinner("Looking for the ring..."):
        results[dataset_name] = pipeline.run(dataset, analyst)

result = results.get(dataset_name)
if result and result.error:
    st.error(result.error)
elif result:
    if result.check.known:
        st.markdown("**Suspects, checked against the data:** " + ", ".join(result.check.known))
    else:
        st.markdown("**No suspects** were named.")
    if result.check.invented:
        st.warning("These names are not in the data, so they were left out: " + ", ".join(result.check.invented))

ring = result.ring if result and not result.error else set()
nodes, edges = draw.graph(dataset, ring)
agraph(nodes, edges, Config(height=600, width=900, directed=False, physics=True))

st.caption("Original app by Michael Lively and team \"The Phi Generation\": 1st place, lablab.ai "
           "\"Phi-2 Technology: 24 Hours Challenge\" (February 2024). This fork by Muhammad Faisal.")
