"""Henry, the graph-maker: the dataset as nodes and edges for streamlit-agraph.

There is one node for each person and one for each context (an accident, an SSN...).
The people in the ring are drawn in red. The original app keyed its nodes by role
label ("Driver 1"), so two different people with the same label became one node.
"""
from streamlit_agraph import Edge, Node

RING_COLOR = "#E53935"
PERSON_COLOR = "#90A4AE"
CONTEXT_COLOR = "#FDD835"


def graph(dataset, ring):
    nodes = [
        Node(id=f"person:{person}", label=person, color=RING_COLOR if person in ring else PERSON_COLOR,
             size=22 if person in ring else 15)
        for person in sorted(dataset.people)
    ]
    nodes += [Node(id=f"context:{context}", label=context, color=CONTEXT_COLOR, shape="square", size=12)
              for context in sorted(dataset.contexts)]
    edges = [Edge(source=f"person:{record.person}", target=f"context:{record.context}",
                  label=record.role if dataset.kind == "claims" else "")
             for record in dataset.records]
    return nodes, edges
