"""A real call to Gemini. Skipped unless RUN_REAL_GEMINI=1 and a key is set.

It uses the free tier. On the free tier Google may use prompts to improve its
products, so it sends only the made-up sample data.
"""
import os

import pytest

from graph_detective import data, llm, pipeline

pytestmark = pytest.mark.skipif(not (os.environ.get("RUN_REAL_GEMINI") and llm.gemini_key()),
                                reason="set RUN_REAL_GEMINI=1 and GEMINI_API_KEY for a real Gemini call")


def test_real_gemini_answers_in_the_schema_and_names_are_checked():
    claims = data.load_sample("fraud.json")
    result = pipeline.run(claims, pipeline.gemini_analyst())
    assert result.error is None, result.error
    assert result.suspects, "Gemini named no suspects"
    assert set(result.check.known) <= claims.people
    print("suspects:", result.suspects, "| invented:", result.check.invented)
