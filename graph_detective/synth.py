"""Test data with an answer key: insurance claims, and bank account holders.

The sample files give their answer away: in fraud.json the ring is Accidents 1-5 and
the lowest person numbers. Generated data uses realistic random names, in a shuffled
order, and adds honest look-alikes, so a method has to read the structure, not labels.

Insurance claims:
- A ring stages 3 to 5 claims. Its core (an organizer, and sometimes a corrupt
  lawyer) is in every staged claim; each other member is in two of them.
- Honest claims have 2 to 4 people who appear once, plus honest lawyers and
  adjusters: a few of them handle most claims, often as the same pair.
- Look-alikes: busy honest professionals, spouses who are in two claims together,
  and innocent victims inside staged claims.

Bank account holders:
- Every holder has their own address, phone, SSN and bank account.
- A ring shares one address and one phone, and two of its members share an SSN.
- Look-alikes: households, who share an address and a phone but nothing else.

The same seed always gives the same data.
"""
import random

FIRST = ["Maya", "Omar", "Lena", "Ravi", "Sofia", "Ethan", "Aisha", "Lucas", "Nora", "Mateo",
         "Zara", "Diego", "Hana", "Leo", "Amara", "Ivan", "Chloe", "Samir", "Grace", "Tariq",
         "Elena", "Kofi", "Mei", "Jonas", "Priya", "Hugo", "Leila", "Felix", "Yara", "Noah",
         "Ines", "Kenji", "Olga", "Rafael", "Nadia", "Theo", "Sana", "Marco", "Freya", "Bilal"]
LAST = ["Chen", "Diaz", "Fox", "Patel", "Rossi", "Khan", "Novak", "Silva", "Okafor", "Berg",
        "Haddad", "Kim", "Moreau", "Lopez", "Tanaka", "Walsh", "Nguyen", "Ivanova", "Mensah", "Costa",
        "Weber", "Singh", "Duarte", "Larsen", "Aziz", "Fischer", "Sato", "Brennan", "Kowalski", "Mbeki"]
INITIALS = "ABCDEFGHJKLMNPRSTVW"
ROLES = ["driver", "passenger", "witness"]


class Names:
    """Unique, realistic names, such as 'Maya K. Chen'. Very large graphs, which only the
    detector reads, get a number at the end once the combinations run out."""

    def __init__(self, rng):
        self.rng, self.used = rng, set()

    def new(self):
        for _ in range(50):
            name = f"{self.rng.choice(FIRST)} {self.rng.choice(INITIALS)}. {self.rng.choice(LAST)}"
            if name not in self.used:
                break
        else:
            name = f"{self.rng.choice(FIRST)} {self.rng.choice(LAST)} {len(self.used)}"
        self.used.add(name)
        return name


def weighted_pick(rng, pool):
    """A few people get most of the work: the i-th person is picked with weight 1/(i+1)."""
    return rng.choices(pool, weights=[1 / (i + 1) for i in range(len(pool))])[0]


def insurance(seed, claims, rings=1):
    """`claims` insurance claims with `rings` planted rings. Returns (tree, truth)."""
    rng, names = random.Random(seed), Names(random.Random(seed + 1))
    lawyers = [names.new() for _ in range(max(2, claims // 15))]
    adjusters = [names.new() for _ in range(max(2, claims // 15))]

    staged, truth_rings, victims = [], [], []
    for _ in range(rings):
        size, count = rng.randint(3, 6), rng.randint(3, 5)
        members = [names.new() for _ in range(size)]
        corrupt_lawyer = rng.random() < 0.5
        ring_claims = [[(members[0], rng.choice(ROLES)),
                        (members[1], "lawyer" if corrupt_lawyer else rng.choice(ROLES))] for _ in range(count)]
        for member in members[2:]:
            for index in rng.sample(range(count), 2):
                ring_claims[index].append((member, rng.choice(ROLES)))
        for index, claim in enumerate(ring_claims):
            if index == 0 or rng.random() < 0.5:
                victims.append(names.new())
                claim.append((victims[-1], rng.choice(["passenger", "witness"])))
        if corrupt_lawyer:
            lawyers.append(members[1])  # also works honest claims
        staged += ring_claims
        truth_rings.append(sorted(members))
    if len(staged) > claims:
        raise ValueError(f"{rings} rings need more than {claims} claims")

    honest = []
    for _ in range(claims - len(staged)):
        claim = [(names.new(), rng.choice(ROLES)) for _ in range(rng.randint(2, 4))]
        if rng.random() < 0.8:
            lawyer = weighted_pick(rng, lawyers)
            claim.append((lawyer, "lawyer"))
            # The busiest lawyer and adjuster are often assigned together.
            adjuster = adjusters[0] if lawyer == lawyers[0] and rng.random() < 0.7 else weighted_pick(rng, adjusters)
            claim.append((adjuster, "adjuster"))
        honest.append(claim)

    spouses = []
    if len(honest) >= 2:
        for _ in range(max(1, claims // 20)):
            couple = [names.new(), names.new()]
            for index in rng.sample(range(len(honest)), 2):
                honest[index] += [(couple[0], "driver"), (couple[1], "passenger")]
            spouses.append(couple)

    all_claims = staged + honest
    rng.shuffle(all_claims)
    numbers = rng.sample(range(1, claims + 1), claims)
    children = []
    for number, claim in zip(numbers, all_claims):
        rng.shuffle(claim)
        children.append({"name": f"Claim {number}", "children": [{"name": n, "role": r} for n, r in claim]})

    ring_members = {member for ring in truth_rings for member in ring}
    appearances = {}
    for claim in honest:
        for name, role in claim:
            if role in ("lawyer", "adjuster"):
                appearances[name] = appearances.get(name, 0) + 1
    busy = sorted(name for name, count in appearances.items() if count >= 2 and name not in ring_members)
    people = sorted({person["name"] for claim in children for person in claim["children"]})
    truth = {"mode": "insurance", "seed": seed, "size": claims, "rings": truth_rings,
             "hard_negatives": {"busy_professional": busy, "spouses": spouses, "victim": sorted(victims)},
             "people": people}
    return {"name": "Claims Center", "children": children}, truth


def identity(seed, holders, rings=1):
    """`holders` bank account holders with `rings` planted rings. Returns (tree, truth)."""
    rng, names = random.Random(seed), Names(random.Random(seed + 1))
    labels = {kind: iter(rng.sample(range(1, holders * 3 + 1), holders * 3))
              for kind in ("Address", "Phone Number", "SSN", "Bank Account")}
    label = lambda kind: f"{kind} {next(labels[kind])}"

    groups, truth_rings, households = [], [], []
    for _ in range(rings):
        members = [names.new() for _ in range(rng.randint(3, 5))]
        address, phone, ssn = label("Address"), label("Phone Number"), label("SSN")
        details = {member: [address, phone] for member in members}
        for index, member in enumerate(members):
            details[member] += [ssn if index < 2 else label("SSN"), label("Bank Account")]
        groups.append(details)
        truth_rings.append(sorted(members))
    for _ in range(max(1, holders // 10)):
        members = [names.new() for _ in range(rng.randint(2, 3))]
        address, phone = label("Address"), label("Phone Number")
        groups.append({member: [address, phone, label("SSN"), label("Bank Account")] for member in members})
        households.append(members)
    placed = sum(len(group) for group in groups)
    if placed > holders:
        raise ValueError(f"{rings} rings and the households need more than {holders} holders")
    for _ in range(holders - placed):
        groups.append({names.new(): [label("Address"), label("Phone Number"), label("SSN"), label("Bank Account")]})

    children = [{"name": holder, "children": [{"name": detail} for detail in rng.sample(details, len(details))]}
                for group in groups for holder, details in group.items()]
    rng.shuffle(children)
    truth = {"mode": "identity", "seed": seed, "size": holders, "rings": truth_rings,
             "hard_negatives": {"household": households},
             "people": sorted(child["name"] for child in children)}
    return {"name": "Bank accounts", "children": children}, truth
