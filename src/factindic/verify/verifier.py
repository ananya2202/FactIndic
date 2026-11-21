"""Combined verification engine: integrating Claim Splitting, Rules, and NLI."""

from typing import Any, Dict, List, Optional
from .claim_splitter import Claim, ClaimSplitter
from .key_facts import KeyFact, KeyFactExtractor
from .nli import NLIVerifier
from .rules import RuleMatcher


class FactualityVerifier:
    """
    Main verification pipeline coordinator.
    Evaluates factual consistency of a summary against the source article.
    """

    def __init__(
        self,
        critical_weight: float = 2.0,
        standard_weight: float = 1.0,
        hallucination_penalty: float = 1.0,
        entailment_threshold: float = 0.65,
        use_hf: bool = False,
    ):
        self.splitter = ClaimSplitter()
        self.rules = RuleMatcher()
        self.nli = NLIVerifier(entailment_threshold=entailment_threshold, use_hf=use_hf)
        self.key_facts_extractor = KeyFactExtractor(
            critical_weight=critical_weight,
            standard_weight=standard_weight,
        )
        self.critical_weight = critical_weight
        self.standard_weight = standard_weight
        self.penalty = hallucination_penalty

    def verify_summary(self, source_text: str, summary_text: str) -> Dict[str, Any]:
        """
        Executes complete verification of summary claims against source text.
        """
        claims = self.splitter.split_claims(summary_text)
        evaluated_claims = []

        unsupported_count = 0
        critical_unsupported_count = 0
        error_distribution = {"number_drift": 0, "date_alteration": 0, "entity_swap": 0, "unsupported_claim": 0, "relation_distortion": 0}

        for c in claims:
            rule_res = self.rules.verify_claim(c.text, source_text)
            nli_res = self.nli.score(source_text, c.text)

            is_contradicted = (nli_res["label"] == "contradiction")
            is_grounded = (nli_res["label"] == "entailment") or (nli_res["label"] == "neutral" and nli_res["entailment_score"] >= 0.25)

            # Combined decision: passes if strict rules pass, claim is grounded, and not contradicted
            if rule_res["passed"] and is_grounded and not is_contradicted:
                verdict = "supported"
                error_type = "none"
            else:
                verdict = "unsupported"
                unsupported_count += 1
                if c.claim_type in ("number", "date", "entity"):
                    critical_unsupported_count += 1

                # Prioritize rule error type if rule failed, else NLI failure
                if not rule_res["passed"]:
                    error_type = rule_res["error_type"]
                elif is_contradicted:
                    error_type = "relation_distortion"
                else:
                    error_type = "unsupported_claim"

                if error_type in error_distribution:
                    error_distribution[error_type] += 1

            evaluated_claims.append({
                "claim_id": c.claim_id,
                "text": c.text,
                "type": c.claim_type,
                "rule_passed": rule_res["passed"],
                "rule_issues": rule_res["issues"],
                "nli_label": nli_res["label"],
                "nli_score": nli_res["entailment_score"],
                "verdict": verdict,
                "error_type": error_type,
            })

        total_claims = len(claims)
        hallucination_rate = (unsupported_count / total_claims) if total_claims > 0 else 0.0

        # Evaluate key facts coverage for Factual Recall
        key_facts = self.key_facts_extractor.extract_key_facts(source_text)
        covered_weight = 0.0
        total_possible_weight = sum(kf.weight for kf in key_facts) if key_facts else 1.0

        for kf in key_facts:
            # Check if key fact elements appear in summary text
            kf_nli = self.nli.score(summary_text, kf.text)
            if kf_nli["entailment_score"] >= 0.50 or any(w in summary_text for w in kf.text.split() if len(w) > 4):
                covered_weight += kf.weight

        factual_recall = min(1.0, covered_weight / total_possible_weight)

        # Weighted Key-Fact Score formula
        weighted_score = max(0.0, (covered_weight / total_possible_weight) - (self.penalty * (critical_unsupported_count / max(1, total_claims))))

        return {
            "total_claims": total_claims,
            "supported_claims": total_claims - unsupported_count,
            "unsupported_claims": unsupported_count,
            "hallucination_rate": round(hallucination_rate, 4),
            "factual_recall": round(factual_recall, 4),
            "weighted_score": round(weighted_score, 4),
            "error_distribution": error_distribution,
            "claims": evaluated_claims,
            "key_facts_count": len(key_facts),
        }
