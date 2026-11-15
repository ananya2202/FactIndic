"""Unit tests for Person C (LLM prompts, caching, and revision loop)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.llm.client import MockLLMClient
from factindic.llm.prompts import (
    FACTUALITY_AWARE_PROMPT,
    REVISION_PROMPT,
    STANDARD_PROMPT,
    format_prompt,
)
from factindic.llm.reviser import SummaryReviser


def test_prompt_formatting():
    input_text = "दिल्ली में प्रदूषण का स्तर 432 पहुंच गया।"
    p_std = format_prompt(STANDARD_PROMPT, input_text=input_text)
    p_fact = format_prompt(FACTUALITY_AWARE_PROMPT, input_text=input_text)

    assert input_text in p_std
    assert "FACTUALITY INSTRUCTIONS" in p_fact
    assert "concise summary" in p_fact


def test_response_cache_determinism(tmp_path):
    from factindic.llm.cache import ResponseCache
    cache_file = str(tmp_path / "test_cache.json")
    cache = ResponseCache(cache_file)
    client = MockLLMClient(cache=cache)

    prompt = "Summarize this:"
    text = "इसरो ने मौसम उपग्रह का सफल प्रक्षेपण किया।"

    # Call 1: uncached
    res1 = client.generate(prompt, text)
    assert res1["cached"] is False
    assert len(res1["text"]) > 0

    # Call 2: must be cached
    res2 = client.generate(prompt, text)
    assert res2["cached"] is True
    assert res2["text"] == res1["text"]


def test_summary_reviser():
    client = MockLLMClient()
    reviser = SummaryReviser(llm_client=client, max_rounds=2)

    source = "केंद्रीय प्रदूषण नियंत्रण बोर्ड ने बताया कि दिल्ली में AQI 432 दर्ज किया गया।"
    prev_summary = "दिल्ली में AQI 432 दर्ज किया गया तथा संबंधित अधिकारियों ने नए आदेश जारी किए।"
    flagged = ["संबंधित अधिकारियों ने नए आदेश जारी किए"]

    rev_res = reviser.revise_summary(
        source_text=source,
        current_summary=prev_summary,
        flagged_claims=flagged,
        current_round=1,
    )

    assert rev_res["revised"] is True
    assert rev_res["round"] == 1
    # Revised summary must not have the hallucinated addition
    assert "नए आदेश" not in rev_res["revised_summary"]
