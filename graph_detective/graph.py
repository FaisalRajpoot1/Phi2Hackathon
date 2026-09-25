"""Sam, the detector: find fraud rings with plain graph rules. No AI, and no key, needed.

Two people are linked when the contexts they share add up to 2 or more: a claim,
a phone number, an address or an email counts 1; an SSN or an account counts 2.
A link between two professionals (a lawyer and an adjuster, say) does not count
on its own, because professionals meet in many claims for honest reasons.
A ring is a connected group of linked people.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

WEIGHTS = {"ssn": 2, "account": 2, "card": 2, "loan": 2}  # every other kind of context counts 1
PROFESSIONALS = {"lawyer", "adjuster", "doctor", "body shop owner"}
MIN_LINK = 2


@dataclass
class Ring:
    members: list   # names, in natural order
    evidence: dict  # (person, person) -> the contexts they share


def natural_key(text):
    """Sort 'Accident 2' before 'Accident 10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def shared_contexts(dataset):
    """Every pair of people who share at least one context, and the contexts they share."""
    people_in = defaultdict(set)
    for record in dataset.records:
        people_in[record.context].add(record.person)
    shared = defaultdict(list)
    for context, people in people_in.items():
        for pair in combinations(sorted(people), 2):
            shared[pair].append(context)
    return shared


def detect(dataset):
    """The rings in a dataset, biggest first."""
    kind_of = {record.context: record.context_kind for record in dataset.records}
    professionals = {record.person for record in dataset.records if record.role in PROFESSIONALS}
    links, evidence = nx.Graph(), {}
    for (a, b), contexts in shared_contexts(dataset).items():
        if a in professionals and b in professionals:
            continue
        if sum(WEIGHTS.get(kind_of[context], 1) for context in contexts) >= MIN_LINK:
            links.add_edge(a, b)
            evidence[(a, b)] = sorted(contexts, key=natural_key)
    rings = [Ring(sorted(members, key=natural_key),
                  {pair: shared for pair, shared in evidence.items() if pair[0] in members})
             for members in nx.connected_components(links)]
    return sorted(rings, key=lambda ring: (-len(ring.members), natural_key(ring.members[0])))
