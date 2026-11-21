"""
Script 05: Factual Verification Engine (Person A - Tasks 6, 7 & 8)
Splits summaries into atomic typed claims (number, date, entity, relation),
runs rule-based matchers and NLI entailment verifier, extracts key facts,
and computes Hallucination Rate, Factual Recall, and Weighted Key-Fact Score.
Outputs data/processed/verdicts.jsonl (handoff for Persons C & D).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.loader import load_jsonl, save_jsonl
from factindic.verify.verifier import FactualityVerifier


def main():
    parser = argparse.ArgumentParser(description="Verify factual consistency of summaries.")
    parser.add_argument("--summaries", default="data/processed/summaries.jsonl", help="Summaries JSONL")
    parser.add_argument("--versions", default="data/processed/versions.jsonl", help="Source versions JSONL")
    parser.add_argument("--output", default="data/processed/verdicts.jsonl", help="Verdicts output JSONL")
    parser.add_argument("--critical_weight", type=float, default=2.0, help="Weight for number/date/entity facts")
    parser.add_argument("--standard_weight", type=float, default=1.0, help="Weight for standard facts")
    parser.add_argument("--penalty", type=float, default=1.0, help="Hallucination penalty factor")
    parser.add_argument("--nli_thresh", type=float, default=0.65, help="NLI entailment cutoff threshold")
    parser.add_argument("--use_hf", action="store_true", help="Enable deep HuggingFace transformer model (slower on CPU)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on summaries to verify")
    args = parser.parse_args()

    summaries_path = PROJECT_ROOT / args.summaries
    versions_path = PROJECT_ROOT / args.versions
    output_path = PROJECT_ROOT / args.output

    # Fallback to pilot summaries if full summaries file isn't generated yet
    if not summaries_path.exists():
        fallback_path = PROJECT_ROOT / "data" / "processed" / "pilot_summaries.jsonl"
        if fallback_path.exists():
            summaries_path = fallback_path

    print(f"[Person A - Step 6-8] Loading versions from {versions_path}...")
    versions = {v["version_id"]: v for v in load_jsonl(str(versions_path))}

    print(f"[Person A - Step 6-8] Loading summaries from {summaries_path}...")
    summaries = load_jsonl(str(summaries_path))
    if args.limit:
        summaries = summaries[:args.limit]

    verifier = FactualityVerifier(
        critical_weight=args.critical_weight,
        standard_weight=args.standard_weight,
        hallucination_penalty=args.penalty,
        entailment_threshold=args.nli_thresh,
        use_hf=args.use_hf,
    )

    verdicts = []
    total_claims = 0
    total_supported = 0
    total_unsupported = 0

    print("Verifying summary claims against source articles...")
    for idx, s in enumerate(summaries):
        version_id = s.get("version_id")
        source_item = versions.get(version_id)
        if not source_item:
            # Try finding by article_id if version_id not exact
            art_id = s.get("article_id")
            source_item = next((v for v in versions.values() if v["article_id"] == art_id), None)

        source_text = source_item["text"] if source_item else s.get("source_text", "")
        summary_text = s.get("summary", "")

        res = verifier.verify_summary(source_text, summary_text)

        total_claims += res["total_claims"]
        total_supported += res["supported_claims"]
        total_unsupported += res["unsupported_claims"]

        verdicts.append({
            "article_id": s.get("article_id"),
            "version_id": version_id,
            "mixing_level": s.get("mixing_level", source_item.get("mixing_target", 0.0) if source_item else 0.0),
            "prompt_type": s.get("prompt_type", "standard"),
            "round": s.get("round", 0),
            "summary": summary_text,
            "source_preview": source_text[:120] + "...",
            "claims": res["claims"],
            "total_claims": res["total_claims"],
            "supported_claims": res["supported_claims"],
            "unsupported_claims": res["unsupported_claims"],
            "hallucination_rate": res["hallucination_rate"],
            "factual_recall": res["factual_recall"],
            "weighted_score": res["weighted_score"],
            "error_distribution": res["error_distribution"],
        })

    save_jsonl(verdicts, str(output_path))
    overall_halluc_rate = (total_unsupported / total_claims) if total_claims > 0 else 0.0

    print(f"\n[Person A - Step 6-8 Complete] Verified {len(verdicts)} summaries containing {total_claims} atomic claims.")
    print(f"Supported claims: {total_supported} ({(total_supported/max(1, total_claims))*100:.1f}%)")
    print(f"Unsupported claims: {total_unsupported} ({(total_unsupported/max(1, total_claims))*100:.1f}%)")
    print(f"Overall Hallucination Rate: {overall_halluc_rate:.4f}")
    print(f"Verdicts handoff saved to: {output_path}")


if __name__ == "__main__":
    main()
