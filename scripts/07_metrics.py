"""Calculate summary-quality and factuality metrics for FactIndic."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "tables" / "metrics_summary.csv"


def load_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def words(text):
    """Count Unicode words, including Hindi and English."""
    return re.findall(r"\w+", str(text or ""), flags=re.UNICODE)


class UnicodeTokenizer:
    """Tokenizer that retains Hindi and other Unicode words."""

    def tokenize(self, text):
        return [word.lower() for word in words(text)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(DATA / "verdicts.jsonl"),
        help="Verdict JSONL file",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT),
        help="Output CSV path",
    )
    args = parser.parse_args()

    verdicts = load_jsonl(Path(args.input))
    versions = load_jsonl(DATA / "versions.jsonl")

    version_map = {
        (row["article_id"], row["version_id"]): row
        for row in versions
    }

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=UnicodeTokenizer(),
    )

    results = []
    missing = 0

    for verdict in verdicts:
        key = (verdict["article_id"], verdict["version_id"])
        version = version_map.get(key)

        if version is None:
            missing += 1
            continue

        reference = version.get("reference_summary", "")
        generated = verdict.get("summary", "")
        source = version.get("text", "")

        if not reference or not generated:
            missing += 1
            continue

        scores = scorer.score(reference, generated)

        summary_word_count = len(words(generated))
        source_word_count = len(words(source))

        compression_ratio = (
            summary_word_count / source_word_count
            if source_word_count else None
        )

        results.append({
            "article_id": verdict["article_id"],
            "version_id": verdict["version_id"],
            "mixing_level": verdict.get("mixing_level"),
            "prompt_type": verdict.get("prompt_type"),
            "round": verdict.get("round"),
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure,
            "bert_score_f1": None,
            "hallucination_rate": verdict.get("hallucination_rate"),
            "factual_recall": verdict.get("factual_recall"),
            "weighted_score": verdict.get("weighted_score"),
            "compression_ratio": compression_ratio,
            "summary_words": summary_word_count,
            "latency_ms": None,
        })

    if not results:
        raise RuntimeError("No valid metric records were generated.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(results)
    frame.to_csv(output, index=False)

    print("\n===== FACTINDIC METRICS =====")
    print("Input verdict records:", len(verdicts))
    print("Successfully processed:", len(frame))
    print("Skipped records:", missing)
    print("Output:", output.relative_to(ROOT) if output.is_relative_to(ROOT) else output)
    print("\nRecords by mixing tier:")
    print(frame.groupby("mixing_level").size().to_string())
    print("\nAverage metrics by mixing tier:")
    print(
        frame.groupby("mixing_level")[
            ["rouge1_f1", "rouge2_f1", "rougeL_f1",
             "hallucination_rate", "factual_recall", "compression_ratio"]
        ].mean().round(4).to_string()
    )


if __name__ == "__main__":
    main()
