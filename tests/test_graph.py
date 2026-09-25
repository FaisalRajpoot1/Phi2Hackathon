"""The detector (Sam): two people are linked when the contexts they share add up to 2 or
more (a claim, phone, address or email counts 1; an SSN or an account counts 2), unless
both are professionals. Linked groups that meet in the same context are one ring.
In claims, a ring has at least 3 people; in bank accounts, it shares an SSN or an account."""
from graph_detective import data, graph
from graph_detective.data import Dataset, Record


def claims(*accidents):
    """A small claims dataset: each accident is a list of (person, role) pairs."""
    records = [Record(f"Claim {n}", "claim", person, role)
               for n, people in enumerate(accidents, start=1) for person, role in people]
    return Dataset("test", "claims", records, lines=[])


def identities(*holders):
    """A small identity dataset: each holder is (name, [(detail, kind), ...])."""
    records = [Record(detail, kind, name, "account holder") for name, details in holders for detail, kind in details]
    return Dataset("test", "identity", records, lines=[])


def flagged(dataset):
    return {person for ring in graph.detect(dataset) for person in ring.members}


def test_the_sample_insurance_ring_is_persons_1_to_6():
    assert flagged(data.load_sample("fraud.json")) == {f"Person {n}" for n in range(1, 7)}


def test_the_sample_banking_ring_is_all_three_holders():
    # Each pair shares the address (1); holders 1 and 2 also share a phone (1),
    # and holders 1 and 3 an SSN (2).
    assert flagged(data.load_sample("fraud2.json")) == {"Account Holder 1", "Account Holder 2", "Account Holder 3"}


def test_three_people_who_keep_meeting_are_a_ring():
    dataset = claims([("Ana", "driver"), ("Ben", "passenger"), ("Cy", "witness")],
                     [("Ana", "witness"), ("Ben", "driver"), ("Cy", "passenger")])
    assert flagged(dataset) == {"Ana", "Ben", "Cy"}


def test_a_couple_in_two_claims_is_not_a_ring():
    dataset = claims([("Ana", "driver"), ("Ben", "passenger")], [("Ana", "driver"), ("Ben", "passenger")])
    assert flagged(dataset) == set()


def test_a_couple_and_their_busy_lawyer_are_not_a_ring():
    # The same busy lawyer handled both of the couple's claims. Lawyers and adjusters
    # can link a ring together, but a ring needs 3 people who are not professionals.
    dataset = claims([("Ana", "driver"), ("Ben", "passenger"), ("Lena Fox", "lawyer")],
                     [("Ana", "driver"), ("Ben", "passenger"), ("Lena Fox", "lawyer")])
    assert flagged(dataset) == set()


def test_a_busy_adjuster_does_not_chain_couples_into_a_ring():
    # One adjuster handles almost every claim. He meets two couples twice each, but
    # nearly everyone else only once, so meeting him proves nothing.
    couples = [[("Ana", "driver"), ("Ben", "passenger")]] * 2 + [[("Cy", "driver"), ("Dee", "passenger")]] * 2
    one_offs = [[(f"Driver {n}", "driver")] for n in range(20)]
    dataset = claims(*[people + [("Omar Diaz", "adjuster")] for people in couples + one_offs])
    assert flagged(dataset) == set()


def test_a_professional_who_keeps_meeting_the_same_people_links_them():
    # Three people meet each other only once, but each meets the same lawyer twice.
    dataset = claims([("Ana", "driver"), ("Lena Fox", "lawyer")], [("Ana", "witness"), ("Lena Fox", "lawyer")],
                     [("Ben", "driver"), ("Lena Fox", "lawyer")], [("Ben", "driver"), ("Lena Fox", "lawyer")],
                     [("Cy", "driver"), ("Lena Fox", "lawyer")], [("Cy", "passenger"), ("Lena Fox", "lawyer")],
                     [("Ana", "passenger"), ("Ben", "witness"), ("Cy", "driver")])
    assert flagged(dataset) == {"Ana", "Ben", "Cy", "Lena Fox"}


def test_linked_groups_that_meet_in_one_claim_are_one_ring():
    # Ana and Ben keep meeting, and so do Cy and Dee; all four are in Claim 1 together.
    dataset = claims([("Ana", "driver"), ("Ben", "passenger"), ("Cy", "witness"), ("Dee", "driver")],
                     [("Ana", "driver"), ("Ben", "passenger")],
                     [("Cy", "driver"), ("Dee", "passenger")])
    assert [ring.members for ring in graph.detect(dataset)] == [["Ana", "Ben", "Cy", "Dee"]]


def test_a_busy_honest_lawyer_is_not_flagged():
    dataset = claims(*[[(f"Driver {n}", "driver"), ("Lena Fox", "lawyer")] for n in range(5)])
    assert flagged(dataset) == set()


def test_a_lawyer_and_adjuster_who_often_meet_are_not_a_ring_on_their_own():
    dataset = claims(*[[(f"Driver {n}", "driver"), ("Lena Fox", "lawyer"), ("Omar Diaz", "adjuster")]
                       for n in range(4)])
    assert flagged(dataset) == set()


def test_sharing_an_ssn_is_enough_on_its_own():
    dataset = identities(("Ana", [("SSN 9", "ssn")]), ("Ben", [("SSN 9", "ssn")]))
    assert flagged(dataset) == {"Ana", "Ben"}


def test_sharing_only_a_phone_is_not():
    # For example, two people in one household.
    dataset = identities(("Ana", [("Phone 5", "phone")]), ("Ben", [("Phone 5", "phone")]))
    assert flagged(dataset) == set()


def test_a_household_sharing_a_home_and_a_phone_is_not_a_ring():
    home = [("Address 3", "address"), ("Phone 5", "phone")]
    dataset = identities(("Ana", home), ("Ben", home), ("Cy", home))
    assert flagged(dataset) == set()


def test_the_evidence_names_the_shared_contexts():
    ring = next(ring for ring in graph.detect(data.load_sample("fraud.json")) if "Person 1" in ring.members)
    assert ring.evidence[("Person 1", "Person 2")] == ["Accident 1", "Accident 3"]
