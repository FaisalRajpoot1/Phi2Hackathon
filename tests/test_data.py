"""Loading the sample files into records, and the compact text the language models read."""
from graph_detective import data


def contexts_of(dataset, person):
    return {record.context for record in dataset.records if record.person == person}


def test_claims_are_keyed_by_person_name():
    claims = data.load_sample("fraud.json")
    assert claims.kind == "claims"
    assert claims.people == {f"Person {n}" for n in range(1, 21)}
    assert claims.contexts == {f"Accident {n}" for n in range(1, 11)}
    assert contexts_of(claims, "Person 4") == {"Accident 1", "Accident 2", "Accident 4"}


def test_role_numbers_are_dropped():
    claims = data.load_sample("fraud.json")
    roles = {record.person: record.role for record in claims.records}
    assert roles["Person 4"] == "lawyer"
    assert roles["Person 6"] == "adjuster"
    assert roles["Person 3"] == "passenger"


def test_people_with_the_same_role_stay_separate():
    # insurance_data.json labels two different people just "Driver". The original app
    # keyed its graph nodes by that label, which merges them into one node.
    claims = data.load_sample("insurance_data.json")
    drivers = {record.person for record in claims.records if record.role == "driver"}
    assert drivers == {"Person 1", "Person 5"}


def test_account_holders_share_the_root_address():
    identity = data.load_sample("fraud2.json")
    assert identity.kind == "identity"
    assert identity.people == {"Account Holder 1", "Account Holder 2", "Account Holder 3"}
    holders_of = lambda context: {r.person for r in identity.records if r.context == context}
    assert holders_of("Address 1") == identity.people
    assert holders_of("SSN 1") == {"Account Holder 1", "Account Holder 3"}
    assert holders_of("Phone Number 1") == {"Account Holder 1", "Account Holder 2"}
    kinds = {record.context: record.context_kind for record in identity.records}
    assert (kinds["Address 1"], kinds["SSN 1"], kinds["Phone Number 1"]) == ("address", "ssn", "phone")


def test_a_root_that_is_not_a_detail_is_not_shared():
    tree = {"name": "Bank accounts", "children": [
        {"name": "Ana", "children": [{"name": "SSN 1"}]},
        {"name": "Ben", "children": [{"name": "SSN 2"}]},
    ]}
    identity = data.load(tree, "identity", "test")
    assert identity.contexts == {"SSN 1", "SSN 2"}


def test_compact_text_has_one_line_per_group():
    lines = data.compact_text(data.load_sample("fraud.json")).splitlines()
    assert len(lines) == 10
    assert lines[0] == "Accident 1: Person 1 [driver]; Person 2 [witness]; Person 4 [lawyer]"


def test_compact_text_for_identities_lists_shared_details():
    lines = data.compact_text(data.load_sample("fraud2.json")).splitlines()
    assert lines[0] == "Account Holder 1: Address 1; Credit Card 1; Bank Account 1; Unsecured Loan 1; Phone Number 1; SSN 1"
