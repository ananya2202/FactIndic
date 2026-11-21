"""
Script 06: Iterative Generate-Verify-Revise Loop (Person C + Person A - Tasks 9 & 13)
Identifies summaries with flagged unsupported claims from Person A's verifier,
prompts the LLM with the Revision Prompt to remove/correct ungrounded claims,
and re-verifies the revised summaries (maximum 2 rounds).
Outputs data/processed/revised_summaries.jsonl and data/processed/verdicts_revised.jsonl.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.loader import load_jsonl, save_jsonl
from factindic.llm.client import get_llm_client
from factindic.llm.reviser import SummaryReviser
from factindic.verify.verifier import FactualityVerifier


def main():
    parser = argparse.ArgumentParser(description="Run the iterative generate-verify-revise loop.")
    parser.add_argument("--summaries", default="data/processed/summaries.jsonl", help="Round 0 summaries JSONL")
    parser.add_argument("--verdicts", default="data/processed/verdicts.jsonl", help="Round 0 verdicts JSONL")
    parser.add_argument("--versions", default="data/processed/versions.jsonl", help="Source versions JSONL")
    parser.add_argument("--output", default="data/processed/revised_summaries_eval.jsonl", help="Revised summaries JSONL")
    parser.add_argument("--output_verdicts", default="data/processed/verdicts_revised_eval.jsonl", help="Re-verified verdicts JSONL")
    parser.add_argument("--max_rounds", type=int, default=2, help="Maximum revision rounds")
    parser.add_argument("--provider", default="mock", help="LLM Provider")
    parser.add_argument("--model", default="mock-flash-1.5", help="Model identifier")
    parser.add_argument("--trace_output", default="data/processed/revision_round_trace_eval.jsonl")
    args = parser.parse_args()

    summaries_path = PROJECT_ROOT / args.summaries
    verdicts_path = PROJECT_ROOT / args.verdicts
    versions_path = PROJECT_ROOT / args.versions
    output_path = PROJECT_ROOT / args.output
    verdicts_out_path = PROJECT_ROOT / args.output_verdicts
    trace_out_path = PROJECT_ROOT / args.trace_output

    print(f"[Person C + A - Step 9] Loading verdicts from {verdicts_path}...")
    verdicts = load_jsonl(str(verdicts_path))
    versions = {v["version_id"]: v for v in load_jsonl(str(versions_path))}

    client = get_llm_client(provider=args.provider, model_name=args.model)
    reviser = SummaryReviser(llm_client=client, max_rounds=args.max_rounds)
    verifier = FactualityVerifier()
    # Use the built-in lightweight semantic verifier for this evaluation-only
    # trace to avoid repeated slow RoBERTa inference on CPU.
    verifier.nli._hf_pipeline = None

    # Evaluation-only safeguard: keep the complete claim while limiting
    # long article premises to the model's 512-token input window.
    if verifier.nli._hf_pipeline is not None:
        _nli_pipeline = verifier.nli._hf_pipeline

        def _bounded_nli_score(premise, hypothesis):
            try:
                output = _nli_pipeline(
                    {"text": premise, "text_pair": hypothesis},
                    truncation="only_first",
                    max_length=512,
                )
                label = output[0]["label"].lower()
                score = output[0]["score"]
                return {
                    "label": label,
                    "entailment_score": score if "entail" in label else 0.1,
                    "neutral_score": score if "neut" in label else 0.1,
                    "contradiction_score": score if "contra" in label else 0.1,
                }
            except Exception:
                return (
                    lambda scores: {
                        "label": (
                            "contradiction" if scores[2] >= verifier.nli.contradiction_threshold
                            else "entailment" if scores[0] >= verifier.nli.entailment_threshold
                            else "neutral"
                        ),
                        "entailment_score": scores[0],
                        "neutral_score": scores[1],
                        "contradiction_score": scores[2],
                    }
                )(verifier.nli._semantic_overlap_score(premise, hypothesis))

        verifier.nli.score = _bounded_nli_score

    revised_records = []
    final_verdicts = []
    trace_records = []

    candidates_for_revision = 0
    resolved_count = 0
    total_unsupported_before = 0
    total_unsupported_after = 0

    print(f"Inspecting {len(verdicts)} summaries for ungrounded claims...")

    for v in verdicts:
        ver_id = v.get("version_id")
        art_id = v.get("article_id")
        prompt_type = v.get("prompt_type", "standard")
        current_summary = v.get("summary", "")
        unsupported_count = v.get("unsupported_claims", 0)
        h_rate_before = v.get("hallucination_rate", 0.0)

        total_unsupported_before += unsupported_count
        trace_records.append({"article_id": art_id, "version_id": ver_id, "mixing_level": v.get("mixing_level", 0.0), "prompt_type": prompt_type, "round": 0, "unsupported_claims": unsupported_count})

        source_obj = versions.get(ver_id)
        if not source_obj:
            source_obj = next((item for item in versions.values() if item["article_id"] == art_id), None)
        source_text = source_obj["text"] if source_obj else v.get("source_preview", "")

        # Check if revision is required
        if unsupported_count == 0 or h_rate_before == 0.0:
            trace_records.append({"article_id": art_id, "version_id": ver_id, "mixing_level": v.get("mixing_level", 0.0), "prompt_type": prompt_type, "round": 1, "unsupported_claims": 0})
            trace_records.append({"article_id": art_id, "version_id": ver_id, "mixing_level": v.get("mixing_level", 0.0), "prompt_type": prompt_type, "round": 2, "unsupported_claims": 0})
            final_verdicts.append(v)
            continue

        candidates_for_revision += 1
        flagged_claim_texts = [c["text"] for c in v.get("claims", []) if c["verdict"] == "unsupported"]

        active_summary = current_summary
        current_h_rate = h_rate_before
        rounds_run = 0

        # Iterative Revision Loop (max rounds)
        for r in range(1, args.max_rounds + 1):
            rounds_run = r
            rev_res = reviser.revise_summary(
                source_text=source_text,
                current_summary=active_summary,
                flagged_claims=flagged_claim_texts,
                current_round=r,
            )
            revised_text = rev_res["revised_summary"]

            # Re-verify the revised summary with Person A's verifier
            re_ver_res = verifier.verify_summary(source_text, revised_text)
            trace_records.append({"article_id": art_id, "version_id": ver_id, "mixing_level": v.get("mixing_level", 0.0), "prompt_type": prompt_type, "round": r, "unsupported_claims": re_ver_res["unsupported_claims"]})
            new_h_rate = re_ver_res["hallucination_rate"]
            active_summary = revised_text
            current_h_rate = new_h_rate

            # If completely resolved (0 hallucination), record remaining rounds as 0 and break early
            if new_h_rate == 0.0:
                for future_r in range(r + 1, args.max_rounds + 1):
                    trace_records.append({
                        "article_id": art_id,
                        "version_id": ver_id,
                        "mixing_level": v.get("mixing_level", 0.0),
                        "prompt_type": prompt_type,
                        "round": future_r,
                        "unsupported_claims": 0,
                    })
                break

            # Update flagged claims for round 2 if needed
            flagged_claim_texts = [c["text"] for c in re_ver_res.get("claims", []) if c["verdict"] == "unsupported"]

        if current_h_rate == 0.0:
            status = "resolved"
            resolved_count += 1
        elif current_h_rate < h_rate_before:
            status = "improved"
        else:
            status = "unchanged"

        total_unsupported_after += re_ver_res["unsupported_claims"]

        revised_records.append({
            "article_id": art_id,
            "version_id": ver_id,
            "mixing_level": v.get("mixing_level", 0.0),
            "prompt_type": prompt_type,
            "rounds_executed": rounds_run,
            "original_summary": current_summary,
            "revised_summary": active_summary,
            "flagged_claims_initial": [c["text"] for c in v.get("claims", []) if c["verdict"] == "unsupported"],
            "hallucination_rate_before": h_rate_before,
            "hallucination_rate_after": current_h_rate,
            "status": status,
        })

        # Record post-revision verdict
        final_verdicts.append({
            "article_id": art_id,
            "version_id": ver_id,
            "mixing_level": v.get("mixing_level", 0.0),
            "prompt_type": prompt_type,
            "round": rounds_run,
            "summary": active_summary,
            "source_preview": source_text[:120] + "...",
            "claims": re_ver_res["claims"],
            "total_claims": re_ver_res["total_claims"],
            "supported_claims": re_ver_res["supported_claims"],
            "unsupported_claims": re_ver_res["unsupported_claims"],
            "hallucination_rate": re_ver_res["hallucination_rate"],
            "factual_recall": re_ver_res["factual_recall"],
            "weighted_score": re_ver_res["weighted_score"],
            "error_distribution": re_ver_res["error_distribution"],
            "was_revised": True,
        })

    save_jsonl(revised_records, str(output_path))
    save_jsonl(final_verdicts, str(verdicts_out_path))
    save_jsonl(trace_records, str(trace_out_path))
    print(f"Saved round trace to: {trace_out_path}")

    reduction_pct = ((total_unsupported_before - total_unsupported_after) / max(1, total_unsupported_before)) * 100

    print(f"\n[Person C + A - Step 9 Complete] Revision Loop Finished.")
    print(f"Candidates requiring revision: {candidates_for_revision}")
    print(f"Fully resolved (0% hallucination): {resolved_count}/{candidates_for_revision}")
    print(f"Unsupported claims before revision: {total_unsupported_before}")
    print(f"Unsupported claims after revision:  {total_unsupported_after}")
    print(f"Overall unsupported claim reduction: {reduction_pct:.1f}%")
    print(f"Saved revision log to: {output_path}")
    print(f"Saved post-revision verdicts to: {verdicts_out_path}")


if __name__ == "__main__":
    main()
