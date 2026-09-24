"""Henry, the graph-maker: one node per person and per context, with the ring in red."""
from graph_detective import data, draw


def labels_of(nodes):
    return [node.label for node in nodes]


def test_one_node_per_person_and_per_context():
    claims = data.load_sample("fraud.json")
    nodes, edges = draw.graph(claims, ring=set())
    ids = [node.id for node in nodes]
    assert len(ids) == len(set(ids))
    assert set(labels_of(nodes)) == claims.people | claims.contexts
    assert len(edges) == len(claims.records)  # one link for each person in each claim


def test_ring_members_are_red_and_no_one_else():
    claims = data.load_sample("fraud.json")
    nodes, _ = draw.graph(claims, ring={"Person 1", "Person 4"})
    red = {node.label for node in nodes if node.color == draw.RING_COLOR}
    assert red == {"Person 1", "Person 4"}
