"""Donna, the verifier: every name an analyst gives is checked against the data."""
from graph_detective import data, verify


def test_known_and_invented_names_are_separated():
    claims = data.load_sample("fraud.json")
    check = verify.check_names(["Person 1", "person 4 ", "Person 99"], claims)
    assert check.known == ["Person 1", "Person 4"]
    assert check.invented == ["Person 99"]


def test_a_name_given_twice_counts_once():
    claims = data.load_sample("fraud.json")
    check = verify.check_names(["Person 2", "Person 2", "PERSON 2"], claims)
    assert check.known == ["Person 2"]
    assert check.invented == []
