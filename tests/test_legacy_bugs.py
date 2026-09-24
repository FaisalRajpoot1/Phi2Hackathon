"""Proof of the original bugs, run against the unchanged code in legacy/ and against the new code.

Each legacy test is marked xfail(strict=True): it must fail, and with the error named in
`raises`. If a legacy test ever passes, the fake stopped reproducing the bug, and the run fails.
"""
import ast
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
from langchain_core.tools import tool

from fakes import REPLY, legacy_modules

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
PHI_TOOL = "crewai/tools/phi2_tools.py"
GEMINI_TOOL = "crewai/tools/gemini_tools.py"
PHI2_CONTEXT = 2048
PROMPT = "Find the fraud ring."


def load_legacy(relpath, monkeypatch):
    """Load one legacy file by its path, with fakes for everything heavy it imports.

    Loading by path skips legacy/crewai/__init__.py, which breaks on today's LangChain.
    """
    modules, record = legacy_modules(tool)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = LEGACY / relpath
    spec = importlib.util.spec_from_file_location(f"legacy_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, record


def run_legacy_phi_tool(monkeypatch):
    """Run the legacy Phi-2 tool to the end, and return the settings it gave model.generate()."""
    legacy, record = load_legacy(PHI_TOOL, monkeypatch)
    list(legacy.Phi2SearchTools.phi2_search.func(PROMPT))
    return record.model.generate_kwargs


# ---- The Phi-2 call returns text ----------------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="The Phi-2 tool uses yield, so the caller gets a generator object, not text")
def test_legacy_phi_tool_returns_text(monkeypatch):
    legacy, _ = load_legacy(PHI_TOOL, monkeypatch)
    result = legacy.Phi2SearchTools.phi2_search.run(PROMPT)
    assert isinstance(result, str) and REPLY in result


def test_phi_returns_text(local_model):
    from graph_detective import llm

    assert llm.local_generate(PROMPT) == REPLY


# ---- Importing the code loads no model ----------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Importing the Phi-2 tool loads the whole model, before anyone asks for it")
def test_legacy_import_loads_no_model(monkeypatch):
    _, record = load_legacy(PHI_TOOL, monkeypatch)
    assert record.loads == 0


def test_import_loads_no_model(local_model):
    from graph_detective import llm

    importlib.reload(llm)
    assert local_model.loads == []


# ---- The answer fits in Phi-2's context ---------------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="The tool asks for 2,048 new tokens, which alone fills Phi-2's 2,048-token context")
def test_legacy_output_fits_phi2_context(monkeypatch):
    settings = run_legacy_phi_tool(monkeypatch)
    assert len(settings["input_ids"][0]) + settings["max_new_tokens"] <= PHI2_CONTEXT


def test_output_fits_phi2_context(local_model):
    from graph_detective import llm

    llm.local_generate(PROMPT)
    call = local_model.model.calls[-1]
    assert call["prompt_tokens"] + call["max_new_tokens"] <= PHI2_CONTEXT


# ---- A temperature is only given when sampling --------------------------

@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="The tool passes temperature=0.75 without do_sample, so the temperature is ignored")
def test_legacy_temperature_only_with_sampling(monkeypatch):
    settings = run_legacy_phi_tool(monkeypatch)
    assert "temperature" not in settings or settings.get("do_sample")


def test_temperature_only_with_sampling(local_model):
    from graph_detective import llm

    llm.local_generate(PROMPT)
    call = local_model.model.calls[-1]
    assert "temperature" not in call or call.get("do_sample")


# ---- The Gemini code imports without a key ------------------------------

@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="The Gemini tool raises ValueError at import when no key is set")
def test_legacy_gemini_imports_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    load_legacy(GEMINI_TOOL, monkeypatch)


def test_gemini_imports_without_key(monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    from graph_detective import llm

    importlib.reload(llm)


# ---- Every module the app imports exists --------------------------------

def normalize(name):
    return re.sub(r"[-_.]+", "_", name).lower()


def requirement_names(path):
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line and not line.startswith("-"):
            names.add(normalize(re.split(r"[<>=!~\[; ]", line)[0]))
    return names


def imported_modules(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def missing_imports(app_path, requirements_path):
    """Modules the app imports that are not in Python, not in the repo and not in its requirements."""
    here = app_path.parent
    required = requirement_names(requirements_path)

    def found(name):
        return (name in sys.stdlib_module_names or (here / f"{name}.py").exists() or (here / name).is_dir()
                or normalize(name) in required or importlib.util.find_spec(name) is not None)

    return {name for name in imported_modules(app_path) if not found(name)}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="app.py imports layout, but layout.py was never added to the GitHub repo")
def test_legacy_app_imports_exist():
    assert missing_imports(LEGACY / "app.py", LEGACY / "requirements.txt") == set()


def test_app_imports_exist():
    assert missing_imports(ROOT / "app.py", ROOT / "requirements.txt") == set()


# ---- Every defined agent runs -------------------------------------------

def agents_defined_and_run(app_path):
    """The names assigned an Agent(...), and the names passed to Crew(agents=[...])."""
    defined, run = set(), set()
    for node in ast.walk(ast.parse(app_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            called = getattr(node.value.func, "id", None)
            if called == "Agent":
                defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
            elif called == "Crew":
                for keyword in node.value.keywords:
                    if keyword.arg == "agents":
                        run.update(element.id for element in keyword.value.elts)
    return defined, run


def test_legacy_app_defines_three_agents():
    defined, _ = agents_defined_and_run(LEGACY / "app.py")
    assert defined == {"Sam", "Donna", "Henry"}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="Three agents are defined, but the crew runs only Sam")
def test_legacy_every_defined_agent_runs():
    defined, run = agents_defined_and_run(LEGACY / "app.py")
    assert run == defined
