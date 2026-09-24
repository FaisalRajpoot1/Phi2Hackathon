"""Show how the original app fails today, one import at a time.

Run it in an environment built from legacy/requirements.txt (today's versions of
those libraries):

    python benchmarks/check_original_imports.py

The Phi-2 tool is left out on purpose: importing it loads the whole model.
"""
import importlib
import sys
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[1] / "legacy"
sys.path.insert(0, str(LEGACY))

STEPS = [
    ("layout, imported by app.py line 7", "layout"),
    ("the bundled crewai, imported by app.py line 21", "crewai"),
    ("LangChain's Gemini connector, imported by app.py line 18", "langchain_google_genai"),
    ("the old Google SDK, imported by app.py line 10", "google.generativeai"),
]

for label, module in STEPS:
    try:
        importlib.import_module(module)
        print(f"OK    {label}")
    except Exception as error:
        print(f"FAIL  {label}: {type(error).__name__}: {error}")
