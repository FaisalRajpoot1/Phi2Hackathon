"""Scoring one method's answer on one generated graph, against its answer key.

Scores are per person: a ring member who was flagged is "correct", one who was not
is "missed", and anyone else who was flagged is a "false flag". A name that is not
in the data at all is also "invented" (and still a false flag). On a graph with no
ring, the only right answer is to flag no one: anything else is a "false alarm".
"""


def names_in(item):
    return item if isinstance(item, list) else [item]


def score(flagged, truth):
    flagged = set(flagged)
    ring_people = {person for ring in truth["rings"] for person in ring}
    correct = len(flagged & ring_people)
    precision = correct / len(flagged) if flagged else None
    recall = correct / len(ring_people) if ring_people else None
    if recall is None:
        f1 = None
    elif not precision or not recall:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    look_alikes = {kind: {name for item in items for name in names_in(item)}
                   for kind, items in truth["hard_negatives"].items()}
    return {
        "correct": correct,
        "missed": len(ring_people - flagged),
        "false_flags": len(flagged - ring_people),
        "invented": len(flagged - set(truth["people"])),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm": None if ring_people else bool(flagged),
        "look_alikes_flagged": {kind: len(names & flagged) for kind, names in look_alikes.items()},
        "look_alikes_total": {kind: len(names) for kind, names in look_alikes.items()},
    }
