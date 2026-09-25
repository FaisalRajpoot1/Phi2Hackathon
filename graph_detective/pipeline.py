"""The three roles, run in order:

1. Sam, the analyst, names the suspects. Sam is either the graph detector (exact,
   and no key needed), or a language model (Gemini, or Phi on this computer), as
   in the original app.
2. Donna, the verifier, checks every name against the data.
3. Henry, the graph-maker, keeps only the checked names as the ring to draw in red.
"""
import re
from dataclasses import dataclass, field

from pydantic import BaseModel

from graph_detective import data, graph, llm, verify


class SuspectList(BaseModel):
    suspects: list[str]


INSTRUCTIONS = {
    "claims": ("Below are insurance claims, one per line: the claim, then each person in it, with their "
               "role in brackets. Find the people who look like part of a fraud ring: the same people "
               "turning up together in different claims."),
    "identity": ("Below are bank account holders, one per line, with the details on their accounts. Find "
                 "the account holders who look like part of a fraud ring: holders who share details that "
                 "should belong to one person only, such as an SSN or a phone number."),
}
ANSWER_FORMAT = ('Answer only with JSON in this form: {"suspects": ["name", ...]}. '
                 "Write each name exactly as it appears below.")


def build_prompt(dataset):
    return f"{INSTRUCTIONS[dataset.kind]} {ANSWER_FORMAT}\n\n{data.compact_text(dataset)}"


def detector_analyst():
    def analyst(dataset):
        return [person for ring in graph.detect(dataset) for person in ring.members]
    return analyst


def gemini_analyst(model=llm.DEFAULT_GEMINI_MODEL, api_key=None):
    def analyst(dataset):
        return llm.gemini_json(build_prompt(dataset), SuspectList, model=model, api_key=api_key).suspects
    return analyst


def local_analyst(model_id=llm.PHI2):
    def analyst(dataset):
        return parse_suspects(llm.local_generate(build_prompt(dataset), model_id=model_id))
    return analyst


def parse_suspects(text):
    """Read {"suspects": [...]} from a model's free text: the first such object wins."""
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            return SuspectList.model_validate_json(match.group()).suspects
        except ValueError:
            continue
    raise llm.BadAnswer("The answer has no JSON list of suspects.", raw=text)


@dataclass
class Result:
    suspects: list = field(default_factory=list)
    check: verify.NameCheck = field(default_factory=lambda: verify.NameCheck([], []))
    ring: set = field(default_factory=set)
    error: str | None = None


def run(dataset, analyst):
    try:
        suspects = list(analyst(dataset))  # Sam
    except Exception as error:  # the page shows the reason, instead of crashing
        return Result(error=str(error))
    check = verify.check_names(suspects, dataset)  # Donna
    return Result(suspects=suspects, check=check, ring=set(check.known))  # Henry
