"""The test-data generator: realistic names, shuffled order, planted rings, honest look-alikes,
and an answer key for every graph."""
import re

from graph_detective import data, graph, synth


def load(tree, kind):
    return data.load(tree, kind, "synthetic")


def people_in(tree):
    return {person["name"] for group in tree["children"] for person in group["children"]}


def test_the_same_seed_gives_the_same_data():
    assert synth.insurance(seed=1, claims=20) == synth.insurance(seed=1, claims=20)
    assert synth.insurance(seed=1, claims=20) != synth.insurance(seed=2, claims=20)


def test_insurance_data_loads_with_the_requested_number_of_claims():
    tree, truth = synth.insurance(seed=3, claims=40)
    claims = load(tree, "claims")
    assert len(claims.contexts) == 40
    assert set(truth["people"]) == claims.people


def test_there_are_no_numbering_hints():
    tree, _ = synth.insurance(seed=4, claims=20)
    assert not any(re.fullmatch(r"Person \d+", name) for name in people_in(tree))
    numbers = [int(claim["name"].split()[-1]) for claim in tree["children"]]
    assert numbers != sorted(numbers)


def test_every_ring_member_shares_two_claims_with_another_member():
    tree, truth = synth.insurance(seed=5, claims=40, rings=2)
    shared = graph.shared_contexts(load(tree, "claims"))
    assert len(truth["rings"]) == 2
    for ring in truth["rings"]:
        for member in ring:
            assert any(len(shared.get(tuple(sorted((member, other))), [])) >= 2 for other in ring if other != member)


def test_a_graph_can_have_no_ring():
    _, truth = synth.insurance(seed=6, claims=10, rings=0)
    assert truth["rings"] == []


def test_honest_look_alikes_are_marked_and_are_not_in_rings():
    tree, truth = synth.insurance(seed=7, claims=40)
    ring_members = {member for ring in truth["rings"] for member in ring}
    look_alikes = truth["hard_negatives"]
    assert look_alikes["busy_professional"] and look_alikes["spouses"] and look_alikes["victim"]
    for names in look_alikes.values():
        flat = {name for item in names for name in (item if isinstance(item, list) else [item])}
        assert not flat & ring_members
    shared = graph.shared_contexts(load(tree, "claims"))
    for a, b in look_alikes["spouses"]:
        assert len(shared[tuple(sorted((a, b)))]) == 2


def test_identity_rings_share_a_strong_detail_and_households_do_not():
    tree, truth = synth.identity(seed=8, holders=30)
    identity = load(tree, "identity")
    kind_of = {record.context: record.context_kind for record in identity.records}
    shared = graph.shared_contexts(identity)
    strong = {"ssn", "account", "card"}
    for ring in truth["rings"]:
        pairs = [tuple(sorted((a, b))) for a in ring for b in ring if a < b]
        assert any(kind_of[context] in strong for pair in pairs for context in shared.get(pair, []))
    for household in truth["hard_negatives"]["household"]:
        pair = tuple(sorted(household[:2]))
        assert {kind_of[context] for context in shared[pair]} == {"address", "phone"}


def test_identity_data_loads_with_the_requested_number_of_holders():
    tree, truth = synth.identity(seed=9, holders=30)
    assert len(load(tree, "identity").people) == 30
