"""
Person A Verification Test & Diagnostic Suite (Task 5 & Task 8)
Tests NLI entailment scoring on IndicXNLI-style premise-hypothesis pairs,
atomic claim splitting, deterministic number/entity/date rule verification,
and weighted scoring.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.verify.claim_splitter import ClaimSplitter
from factindic.verify.key_facts import KeyFactExtractor
from factindic.verify.nli import NLIVerifier
from factindic.verify.rules import RuleMatcher
from factindic.verify.verifier import FactualityVerifier


def test_nli_indic_pairs():
    print("=" * 70)
    print("1. Person A Task 5: Testing Indic NLI Entailment Scoring")
    print("=" * 70)
    nli = NLIVerifier(entailment_threshold=0.60)

    test_cases = [
        {
            "desc": "Direct Entailment (Hindi)",
            "premise": "केंद्रीय प्रदूषण नियंत्रण बोर्ड ने बताया कि दिल्ली में AQI 432 दर्ज किया गया।",
            "hypothesis": "दिल्ली में AQI 432 पहुंच गया।",
            "expected": "entailment",
        },
        {
            "desc": "Code-Mixed Entailment (Hinglish)",
            "premise": "ISRO successfully launched the weather satellite INSAT-3DS from Sriharikota.",
            "hypothesis": "ISRO ने weather satellite successfully launch किया।",
            "expected": "entailment",
        },
        {
            "desc": "Direct Contradiction (Number Drift)",
            "premise": "मृतकों की संख्या 12 बताई गई है।",
            "hypothesis": "हादसे में 50 लोगों की मौत हो गई।",
            "expected": "contradiction",
        },
        {
            "desc": "Neutral / Extrapolation (Unsupported assertion)",
            "premise": "कंपनी ने गुजरात में नया कारखाना लगाने की घोषणा की।",
            "hypothesis": "कंपनी अगले साल 1 लाख नई नौकरियां देगी।",
            "expected": "neutral",
        },
    ]

    for tc in test_cases:
        res = nli.score(tc["premise"], tc["hypothesis"])
        status = "PASS" if res["label"] == tc["expected"] else "FLAGGED"
        print(f"[{status}] {tc['desc']}")
        print(f"  Premise:    {tc['premise']}")
        print(f"  Hypothesis: {tc['hypothesis']}")
        print(f"  Result:     Label={res['label']} (P_entail={res['entailment_score']}, P_contra={res['contradiction_score']})\n")


def test_typed_claim_splitter():
    print("=" * 70)
    print("2. Person A Task 8: Testing Typed-Claim Splitter on Code-Mixed Summary")
    print("=" * 70)
    splitter = ClaimSplitter()
    sample_summary = (
        "दिल्ली में शुक्रवार को AQI 432 पहुंच गया, CPCB ने ग्रैप-4 लागू किया "
        "और प्राथमिक स्कूलों को 15 नवंबर तक बंद करने का आदेश दिया।"
    )
    claims = splitter.split_claims(sample_summary)
    print(f"Summary input: {sample_summary}\n")
    print(f"Extracted {len(claims)} atomic claims:")
    for c in claims:
        print(f"  - [{c.claim_type.upper()}] Claim {c.claim_id}: \"{c.text}\"")
    print()


def test_rule_matchers():
    print("=" * 70)
    print("3. Person A Task 8: Testing Deterministic Rule Verification")
    print("=" * 70)
    rules = RuleMatcher()
    source_text = "केंद्रीय इलेक्ट्रॉनिक्स मंत्रालय ने गुजरात में 22,500 करोड़ रुपये के निवेश से सेमीकंडक्टर प्लांट को मंजूरी दी।"

    # Case 1: Faithful claim
    claim1 = "गुजरात में 22,500 करोड़ रुपये का निवेश होगा"
    res1 = rules.verify_claim(claim1, source_text)
    print(f"Claim 1: \"{claim1}\"")
    print(f"  Passed: {res1['passed']} (Issues: {res1['issues']})\n")

    # Case 2: Number drift
    claim2 = "गुजरात में 50,000 करोड़ रुपये का निवेश होगा"
    res2 = rules.verify_claim(claim2, source_text)
    print(f"Claim 2 (Number Drift): \"{claim2}\"")
    print(f"  Passed: {res2['passed']} (Error: {res2['error_type']}, Issues: {res2['issues']})\n")

    # Case 3: Entity swap
    claim3 = "महाराष्ट्र में सेमीकंडक्टर प्लांट को मंजूरी दी"
    res3 = rules.verify_claim(claim3, source_text)
    print(f"Claim 3 (Entity Swap): \"{claim3}\"")
    print(f"  Passed: {res3['passed']} (Error: {res3['error_type']}, Issues: {res3['issues']})\n")


def test_end_to_end_verifier():
    print("=" * 70)
    print("4. Person A Task 8 & 14: End-to-End Verifier on Faithful vs Hallucinated Summaries")
    print("=" * 70)
    verifier = FactualityVerifier(critical_weight=2.0, standard_weight=1.0, hallucination_penalty=1.0)

    source = (
        "अखिल भारतीय आयुर्विज्ञान संस्थान (AIIMS) नई दिल्ली ने 45 करोड़ रुपये की लागत से "
        "रोबोटिक सर्जरी केंद्र का उद्घाटन किया है। सर्जरी से मरीज केवल 3 दिन में डिस्चार्ज हो सकेंगे।"
    )

    faithful_summary = "AIIMS ने 45 करोड़ की लागत से रोबोटिक केंद्र शुरू किया और मरीज 3 दिन में ठीक होंगे।"
    res_f = verifier.verify_summary(source, faithful_summary)
    print("Faithful Summary Verification:")
    print(f"  Hallucination Rate: {res_f['hallucination_rate']}")
    print(f"  Factual Recall:     {res_f['factual_recall']}")
    print(f"  Weighted Score:     {res_f['weighted_score']}")
    print(f"  Supported Claims:   {res_f['supported_claims']}/{res_f['total_claims']}\n")

    drifted_summary = "AIIMS ने 90 करोड़ की लागत से केंद्र शुरू किया और मरीज 10 दिन में डिस्चार्ज होंगे।"
    res_d = verifier.verify_summary(source, drifted_summary)
    print("Drifted Summary Verification (Simulated Hallucination):")
    print(f"  Hallucination Rate: {res_d['hallucination_rate']}")
    print(f"  Factual Recall:     {res_d['factual_recall']}")
    print(f"  Weighted Score:     {res_d['weighted_score']}")
    print(f"  Errors Flagged:     {res_d['error_distribution']}")
    print("=" * 70)


if __name__ == "__main__":
    test_nli_indic_pairs()
    test_typed_claim_splitter()
    test_rule_matchers()
    test_end_to_end_verifier()
