"""The fraud datasets, loaded into simple records.

Every sample file is a tree: a root, then groups, then items. There are two kinds:
- "claims": the root is a claims center, the groups are accidents, and the items
  are the people in each accident, with their role.
- "identity": the root is an address that all the account holders share, the
  groups are the account holders, and the items are their details (SSN, phone...).

Whatever the kind, a record says one thing: this person is linked to this context
(an accident, an SSN, a phone number...). People are keyed by name, never by role.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# The sample files, and the kind of each one.
SAMPLES = {
    "fraud.json": "claims",
    "neo4jdata.json": "claims",
    "insurance_data.json": "claims",
    "fraud2.json": "identity",
}

# What an identity detail is, from its label: "Phone Number 1" is a "phone".
DETAIL_KINDS = {
    "address": "address", "ssn": "ssn", "phone number": "phone", "email": "email",
    "credit card": "card", "bank account": "account", "unsecured loan": "loan",
}


@dataclass(frozen=True)
class Record:
    context: str       # "Accident 3", "SSN 1", "Address 1"
    context_kind: str  # "claim", "ssn", "phone", "address"...
    person: str        # "Person 4", "Account Holder 2"
    role: str          # "lawyer", "driver", "account holder"


@dataclass
class Dataset:
    name: str
    kind: str
    records: list
    lines: list  # (group, [item texts]) in file order: what the language models read

    @property
    def people(self):
        return {record.person for record in self.records}

    @property
    def contexts(self):
        return {record.context for record in self.records}


def base_label(label):
    """'Driver 1' -> 'driver', 'Phone Number 12' -> 'phone number'."""
    return re.sub(r"\s*\d+$", "", label.strip()).lower()


def load_claims(tree, name="claims"):
    records, lines = [], []
    for accident in tree["children"]:
        items = []
        for person in accident["children"]:
            role = base_label(person.get("role", ""))
            records.append(Record(accident["name"], "claim", person["name"], role))
            items.append(f"{person['name']} [{role}]")
        lines.append((accident["name"], items))
    return Dataset(name, "claims", records, lines)


def load_identity(tree, name="identity"):
    # In fraud2.json the root is "Address 1", an address all the holders share. A root
    # that is not an identity detail (for example "Bank accounts") is only a title.
    shared = [tree["name"]] if base_label(tree["name"]) in DETAIL_KINDS else []
    records, lines = [], []
    for holder in tree["children"]:
        details = shared + [item["name"] for item in holder["children"]]
        for detail in details:
            kind = DETAIL_KINDS.get(base_label(detail), base_label(detail))
            records.append(Record(detail, kind, holder["name"], "account holder"))
        lines.append((holder["name"], details))
    return Dataset(name, "identity", records, lines)


LOADERS = {"claims": load_claims, "identity": load_identity}


def load(tree, kind, name):
    return LOADERS[kind](tree, name)


def load_sample(filename):
    tree = json.loads((DATA_DIR / filename).read_text(encoding="utf-8"))
    return load(tree, SAMPLES[filename], filename)


def compact_text(dataset):
    """One line per group, for example 'Accident 1: Person 1 [driver]; Person 2 [witness]'."""
    return "\n".join(f"{group}: {'; '.join(items)}" for group, items in dataset.lines)
