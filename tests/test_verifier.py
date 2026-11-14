"""Unit tests for FactualityVerifier (Person A)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.verify.verifier import FactualityVerifier


def test_faithful_summary_scores():
    verifier = FactualityVerifier()
    source = "आरबीआई गवर्नर शक्तिकांत दास ने नीतिगत रेपो दर को 6.50% पर अपरिवर्तित रखने की घोषणा की।"
    summary = "आरबीआई ने रेपो दर 6.50% पर स्थिर रखी।"

    res = verifier.verify_summary(source, summary)
    assert res["hallucination_rate"] == 0.0
    assert res["supported_claims"] == res["total_claims"]
    assert res["weighted_score"] > 0.0


def test_hallucinated_summary_penalty():
    verifier = FactualityVerifier()
    source = "आरबीआई गवर्नर शक्तिकांत दास ने नीतिगत रेपो दर को 6.50% पर अपरिवर्तित रखने की घोषणा की।"
    # 6.50% drifted to 9.25%
    summary = "आरबीआई ने रेपो दर बढ़ाकर 9.25% कर दी।"

    res = verifier.verify_summary(source, summary)
    assert res["hallucination_rate"] > 0.0
    assert res["unsupported_claims"] > 0
    assert "number_drift" in res["error_distribution"] or res["error_distribution"]["unsupported_claim"] > 0
