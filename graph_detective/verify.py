"""Donna, the verifier: every name an analyst gives is checked against the data.

Names match without caring about upper or lower case, or extra spaces. A name
that is not in the data is "invented": the analyst made it up.
"""
from dataclasses import dataclass


@dataclass
class NameCheck:
    known: list     # names found in the data, written the way the data writes them
    invented: list  # names not in the data, written the way the analyst wrote them


def normalize(name):
    return " ".join(name.split()).casefold()


def check_names(names, dataset):
    people = {normalize(person): person for person in dataset.people}
    known, invented, seen = [], [], set()
    for name in names:
        key = normalize(name)
        if key in seen:
            continue
        seen.add(key)
        if key in people:
            known.append(people[key])
        else:
            invented.append(" ".join(name.split()))
    return NameCheck(known, invented)
