"""Iterative revision engine driven by flagged verifier claims."""

from typing import Any, Dict, List, Optional
from .client import BaseLLMClient
from .prompts import REVISION_PROMPT, format_prompt


class SummaryReviser:
    """
    Orchestrates the generate-verify-revise feedback loop.
    Iteratively prompts the LLM to rewrite summaries by stripping or correcting
    claims flagged as unsupported by Person A's verification engine (max 2 rounds).
    """

    def __init__(self, llm_client: BaseLLMClient, max_rounds: int = 2):
        self.client = llm_client
        self.max_rounds = max_rounds

    def revise_summary(
        self,
        source_text: str,
        current_summary: str,
        flagged_claims: List[str],
        current_round: int = 1,
    ) -> Dict[str, Any]:
        """
        Executes one revision step on an unfaithful summary.
        """
        if not flagged_claims or current_round > self.max_rounds:
            return {
                "revised_summary": current_summary,
                "round": current_round,
                "revised": False,
                "notes": "No revision needed or max rounds reached.",
            }

        formatted_claims = "\n".join(f"- {c}" for c in flagged_claims)
        prompt = format_prompt(
            REVISION_PROMPT,
            source_text=source_text,
            previous_summary=current_summary,
            flagged_claims=formatted_claims,
        )

        gen_result = self.client.generate(prompt=prompt, text="", force_refresh=False)
        revised_text = gen_result["text"].strip()

        return {
            "revised_summary": revised_text,
            "round": current_round,
            "revised": True,
            "flagged_claims": flagged_claims,
            "latency_ms": gen_result.get("latency_ms", 0),
        }

    def revise(
        self,
        source_text: str,
        initial_summary: str,
        flagged_claims: List[str],
        verifier: Any = None,
    ) -> Dict[str, Any]:
        """Multi-round revision wrapper with verifier feedback."""
        active_summary = initial_summary
        rounds_executed = 0
        h_rate = 1.0
        unsupported_count = len(flagged_claims)
        status = "unresolved"

        for r in range(1, self.max_rounds + 1):
            rounds_executed = r
            step_res = self.revise_summary(source_text, active_summary, flagged_claims, current_round=r)
            active_summary = step_res["revised_summary"]
            if verifier:
                v_res = verifier.verify_summary(source_text, active_summary)
                h_rate = v_res.get("hallucination_rate", 0.0)
                unsupported_count = v_res.get("unsupported_claims", 0)
                if h_rate == 0.0:
                    status = "resolved"
                    break
                flagged_claims = [c["text"] for c in v_res.get("claims", []) if c.get("verdict") == "unsupported"]
            else:
                status = "revised"
                break

        return {
            "final_summary": active_summary,
            "final_hallucination_rate": h_rate,
            "final_unsupported_count": unsupported_count,
            "rounds_executed": rounds_executed,
            "status": status,
        }
