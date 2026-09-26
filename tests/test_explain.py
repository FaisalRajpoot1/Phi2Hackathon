"""The AI explains the rings the detector found, and Donna checks every reason against the data."""
from graph_detective import data, explain, graph

CLAIMS = data.load_sample("fraud.json")
RINGS = graph.detect(CLAIMS)


def reason(people, shared, why="They keep turning up together."):
    return explain.Reason(people=people, shared=shared, why=why)


def test_a_true_reason_is_kept():
    kept, rejected = explain.check_reasons([reason(["Person 1", "Person 2"], "Accident 1")], CLAIMS)
    assert [r.shared for r in kept] == ["Accident 1"] and rejected == []


def test_a_person_who_is_not_in_the_data_is_rejected():
    kept, rejected = explain.check_reasons([reason(["Person 1", "Person 99"], "Accident 1")], CLAIMS)
    assert kept == [] and len(rejected) == 1


def test_a_context_that_does_not_exist_is_rejected():
    kept, rejected = explain.check_reasons([reason(["Person 1", "Person 2"], "Accident 42")], CLAIMS)
    assert kept == [] and len(rejected) == 1


def test_a_person_who_is_not_in_the_named_claim_is_rejected():
    # Person 7 is in the data, but not in Accident 1.
    kept, rejected = explain.check_reasons([reason(["Person 1", "Person 7"], "Accident 1")], CLAIMS)
    assert kept == [] and len(rejected) == 1


def test_names_are_matched_without_case_or_extra_spaces():
    kept, _ = explain.check_reasons([reason(["person 1 ", "PERSON 2"], " accident 1")], CLAIMS)
    assert len(kept) == 1


def test_the_prompt_gives_each_ring_and_its_shared_claims():
    prompt = explain.build_prompt(CLAIMS, RINGS)
    assert "Person 1, Person 2, Person 3, Person 4, Person 5, Person 6" in prompt
    assert "Person 1 and Person 2 share: Accident 1, Accident 3" in prompt


def test_explain_returns_checked_reasons_and_counts_the_invented_ones():
    answer = explain.Reasons(reasons=[reason(["Person 1", "Person 2"], "Accident 3"),
                                      reason(["Person 1", "Person 9"], "Accident 3")])
    result = explain.explain(CLAIMS, RINGS, ask=lambda prompt, schema: answer)
    assert [r.shared for r in result.kept] == ["Accident 3"]
    assert len(result.rejected) == 1 and result.error is None


def test_an_ai_error_is_reported_not_raised():
    def failing(prompt, schema):
        raise RuntimeError("The GPU quota is used up.")

    result = explain.explain(CLAIMS, RINGS, ask=failing)
    assert result.error == "The GPU quota is used up." and result.kept == []


def test_a_local_model_answer_is_read_from_free_text():
    text = 'Here you go: {"reasons": [{"people": ["Person 1", "Person 2"], "shared": "Accident 1", "why": "x"}]} Done.'
    assert explain.read_reasons(text).reasons[0].shared == "Accident 1"
