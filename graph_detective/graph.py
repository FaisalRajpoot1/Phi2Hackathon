"""Sam, the detector: find fraud rings with plain graph rules. No AI, and no key, needed.

1. Two people are linked when the contexts they share add up to 2 or more: a claim,
   a phone number, an address or an email counts 1; an SSN or an account counts 2.
   A link between two professionals (a lawyer and an adjuster, say) does not count
   on its own, because professionals meet in many claims for honest reasons. Nor do
   the links of a busy professional who meets almost everyone only once: an honest
   adjuster meets new people all the time, while a corrupt one keeps working with
   the same few (at least 20% of the people they meet, they meet again).
2. Linked groups that meet in the same context (for example, one staged accident)
   are one ring.
3. Honest look-alikes are dropped: in claims, a ring needs at least 3 people who are
   not professionals, since two people in two accidents together is common (a couple,
   often with the same busy lawyer); in bank accounts, a ring needs a shared SSN or
   account, since families share a home and a phone.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import networkx as nx

STRONG = {"ssn", "account", "card", "loan"}  # details that only one person should have
WEIGHTS = {kind: 2 for kind in STRONG}        # every other kind of context counts 1
PROFESSIONALS = {"lawyer", "adjuster", "doctor", "body shop owner"}
MIN_LINK = 2
MIN_CLAIMS_RING = 3
MIN_REPEAT_SHARE = 0.2


@dataclass
class Ring:
    members: list   # names, in natural order
    evidence: dict  # (person, person) -> the contexts they share


def natural_key(text):
    """Sort 'Accident 2' before 'Accident 10'."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def people_by_context(dataset):
    people_in = defaultdict(set)
    for record in dataset.records:
        people_in[record.context].add(record.person)
    return people_in


def shared_contexts(dataset):
    """Every pair of people who share at least one context, and the contexts they share."""
    shared = defaultdict(list)
    for context, people in people_by_context(dataset).items():
        for pair in combinations(sorted(people), 2):
            shared[pair].append(context)
    return shared


def busy_professionals(shared, professionals):
    """Professionals who meet again fewer than MIN_REPEAT_SHARE of the people they meet."""
    met, met_again = defaultdict(set), defaultdict(set)
    for (a, b), contexts in shared.items():
        for professional, other in ((a, b), (b, a)):
            if professional in professionals and other not in professionals:
                met[professional].add(other)
                if len(contexts) >= 2:
                    met_again[professional].add(other)
    return {p for p in met if len(met_again[p]) < MIN_REPEAT_SHARE * len(met[p])}


def is_ring(kind, members, evidence, kind_of, professionals):
    if kind == "identity":
        return any(kind_of[context] in STRONG for contexts in evidence.values() for context in contexts)
    return len(members - professionals) >= MIN_CLAIMS_RING


def detect(dataset):
    """The rings in a dataset, biggest first."""
    kind_of = {record.context: record.context_kind for record in dataset.records}
    professionals = {record.person for record in dataset.records if record.role in PROFESSIONALS}

    # 1. Links between pairs of people.
    shared = shared_contexts(dataset)
    busy = busy_professionals(shared, professionals)
    links, evidence = nx.Graph(), {}
    for (a, b), contexts in shared.items():
        if (a in professionals and b in professionals) or a in busy or b in busy:
            continue
        if sum(WEIGHTS.get(kind_of[context], 1) for context in contexts) >= MIN_LINK:
            links.add_edge(a, b)
            evidence[(a, b)] = sorted(contexts, key=natural_key)

    # 2. Linked groups that meet in the same context become one ring.
    group_of = {person: number for number, members in enumerate(nx.connected_components(links))
                for person in members}
    groups = nx.Graph()
    groups.add_nodes_from(set(group_of.values()))
    for people in people_by_context(dataset).values():
        met = sorted({group_of[person] for person in people if person in group_of})
        groups.add_edges_from(zip(met, met[1:]))

    # 3. Keep the rings; drop honest look-alikes.
    rings = []
    for numbers in nx.connected_components(groups):
        members = {person for person, number in group_of.items() if number in numbers}
        ring_evidence = {pair: shared for pair, shared in evidence.items() if pair[0] in members}
        if is_ring(dataset.kind, members, ring_evidence, kind_of, professionals):
            rings.append(Ring(sorted(members, key=natural_key), ring_evidence))
    return sorted(rings, key=lambda ring: (-len(ring.members), natural_key(ring.members[0])))
