"""
Script 02: English Translation & Alignment Check (Person B - Task 3)
Translates cleaned Hindi articles and reference summaries to English,
verifying that numbers, percentages, dates, and named entities are preserved.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.loader import load_jsonl, save_jsonl
from factindic.data.translator import translate_hindi_to_english, verify_translation_alignment


def main():
    parser = argparse.ArgumentParser(description="Translate Hindi articles to English with alignment checks.")
    parser.add_argument("--input", default="data/processed/clean_articles.jsonl", help="Cleaned articles JSONL")
    parser.add_argument("--output", default="data/processed/translations.jsonl", help="Translated articles JSONL")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    print(f"[Person B - Step 3] Loading cleaned articles from {input_path}...")
    articles = load_jsonl(str(input_path))

    translated_records = []
    total_alignment_issues = 0

    for art in articles:
        art_id = art["id"]
        hi_text = art["text"]
        hi_summary = art.get("summary", "")

        tr_data = translate_hindi_to_english(art_id, hi_text, hi_summary)
        en_text = tr_data["text"]
        en_summary = tr_data["summary"]

        # Run verification check on numbers and entities
        aligned, issues = verify_translation_alignment(hi_text, en_text)
        if not aligned:
            total_alignment_issues += len(issues)
            print(f"  [Warning] Article {art_id} alignment notices: {issues}")

        translated_records.append({
            "article_id": art_id,
            "title_hi": art.get("title", ""),
            "text_hi": hi_text,
            "summary_hi": hi_summary,
            "text_en": en_text,
            "summary_en": en_summary,
            "aligned": aligned,
            "alignment_issues": issues,
            "word_count_hi": len(hi_text.split()),
            "word_count_en": len(en_text.split()),
        })

    save_jsonl(translated_records, str(output_path))
    print(f"[Person B - Step 3 Complete] Processed {len(translated_records)} translations.")
    print(f"Total alignment discrepancies detected: {total_alignment_issues}")
    print(f"Saved translated corpus to {output_path}")


if __name__ == "__main__":
    main()
