"""The AI explains the rings that the detector found, and Donna checks every reason.

The model gets each ring and the contexts its members share, and answers in JSON:
{"reasons": [{"people": ["A", "B"], "shared": "Claim 7", "why": "..."}]}.
Donna keeps a reason only if every person named is in the data, the shared context
exists, and every person named really is in that context. Any other reason is
"rejected": the model made it up. Only the checked reasons are shown.
"""
import json
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from graph_detective import llm
from graph_detective.graph import natural_key
from graph_detective.verify import normalize

EXPLAIN_TOKENS = 512
WORDS = {"claims": ("insurance claims", "claim"), "identity": ("bank accounts", "detail")}


class Reason(BaseModel):
    people: list[str]
    shared: str
    why: str


class Reasons(BaseModel):
    reasons: list[Reason]


@dataclass
class Explanation:
    kept: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    error: str | None = None


def build_prompt(dataset, rings):
    what, context = WORDS[dataset.kind]
    lines = []
    for number, ring in enumerate(rings, start=1):
        lines.append(f"Ring {number}: {', '.join(ring.members)}")
        for (a, b), shared in sorted(ring.evidence.items(), key=lambda item: natural_key(" ".join(item[0]))):
            lines.append(f"- {a} and {b} share: {', '.join(shared)}")
    return (f"Below are fraud rings that were found in {what}, with the {context}s their members share. "
            "For each ring, explain in plain words why it looks like fraud. Give each reason as the people "
            f"it is about and one {context} they share, written exactly as below. Answer only with JSON "
            'in this form: {"reasons": [{"people": ["name", ...], "shared": "...", "why": "..."}]}.\n\n'
            + "\n".join(lines))


def check_reasons(reasons, dataset):
    """Donna's check. Returns (kept, rejected); kept reasons use the data's own spelling."""
    people = {normalize(person): person for person in dataset.people}
    contexts = {normalize(context): context for context in dataset.contexts}
    members = defaultdict(set)
    for record in dataset.records:
        members[record.context].add(record.person)
    kept, rejected = [], []
    for reason in reasons:
        names = [people.get(normalize(name)) for name in reason.people]
        context = contexts.get(normalize(reason.shared))
        if names and None not in names and context and all(name in members[context] for name in names):
            kept.append(Reason(people=names, shared=context, why=reason.why))
        else:
            rejected.append(reason)
    return kept, rejected


def explain(dataset, rings, ask):
    """Ask a model to explain the rings; `ask(prompt, schema)` returns a Reasons object."""
    try:
        answer = ask(build_prompt(dataset, rings), Reasons)
    except Exception as error:  # the page shows the reason, instead of crashing
        return Explanation(error=str(error))
    kept, rejected = check_reasons(answer.reasons, dataset)
    return Explanation(kept, rejected)


def read_reasons(text):
    """Read {"reasons": [...]} from a model's free text: the first such object wins."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char == "{":
            try:
                return Reasons.model_validate(decoder.raw_decode(text, index)[0])
            except (ValueError, ValidationError):
                continue
    raise llm.BadAnswer("The answer has no JSON list of reasons.", raw=text)


def gemini_asker(model=llm.DEFAULT_GEMINI_MODEL, api_key=None):
    return lambda prompt, schema: llm.gemini_json(prompt, schema, model=model, api_key=api_key)


def local_asker(model_id=llm.PHI4_MINI):
    return lambda prompt, schema: read_reasons(llm.local_generate(prompt, model_id=model_id,
                                                                  max_new_tokens=EXPLAIN_TOKENS))
