"""Unit tests for deterministic rule matchers."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.verify.rules import RuleMatcher


def test_rule_number_matching():
    rules = RuleMatcher()
    src = "इसरो ने 2275 किलोग्राम वजनी इनसैट-3डीएस उपग्रह का सफल प्रक्षेपण किया।"

    # Case 1: Exact number matches
    res1 = rules.verify_claim("उपग्रह का वजन 2275 किलोग्राम है", src)
    assert res1["passed"] is True

    # Case 2: Number drift
    res2 = rules.verify_claim("उपग्रह का वजन 5000 किलोग्राम है", src)
    assert res2["passed"] is False
    assert res2["error_type"] == "number_drift"


def test_rule_entity_matching():
    rules = RuleMatcher()
    src = "AIIMS नई दिल्ली ने रोबोटिक सर्जरी केंद्र का उद्घाटन किया।"

    # Case 1: Correct entity
    res1 = rules.verify_claim("AIIMS ने नया केंद्र शुरू किया", src)
    assert res1["passed"] is True

    # Case 2: Entity swap
    res2 = rules.verify_claim("JIPMER ने नया केंद्र शुरू किया", src)
    assert res2["passed"] is False
    assert res2["error_type"] == "entity_swap"
