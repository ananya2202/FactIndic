"""
Script 01: Sample and Clean Articles (Person B - Tasks 1 & 2)
Samples XL-Sum Hindi articles and applies Unicode normalization, zero-width space stripping,
and word-length filtering.
"""

import argparse
import sys
from pathlib import Path

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.cleaner import clean_text, deduplicate_articles, filter_article
from factindic.data.loader import load_jsonl, save_jsonl


def main():
    parser = argparse.ArgumentParser(description="Sample and clean Hindi XL-Sum articles.")
    parser.add_argument("--input", default="data/raw/xlsum_hindi_sample.jsonl", help="Raw input JSONL")
    parser.add_argument("--output", default="data/processed/clean_articles.jsonl", help="Cleaned output JSONL")
    parser.add_argument("--min_words", type=int, default=30, help="Minimum article words")
    parser.add_argument("--max_words", type=int, default=1500, help="Maximum article words")
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    output_path = PROJECT_ROOT / args.output

    print(f"[Person B - Step 1/2] Loading raw articles from {input_path}...")
    raw_articles = load_jsonl(str(input_path))
    print(f"Loaded {len(raw_articles)} raw articles.")

    # Deduplicate
    deduped = deduplicate_articles(raw_articles)
    print(f"Articles after deduplication: {len(deduped)}")

    # Clean and filter
    cleaned_articles = []
    dropped_count = 0

    for art in deduped:
        art_clean = dict(art)
        art_clean["title"] = clean_text(art.get("title", ""))
        art_clean["text"] = clean_text(art.get("text", ""))
        art_clean["summary"] = clean_text(art.get("summary", ""))

        if filter_article(art_clean, min_article_words=args.min_words, max_article_words=args.max_words):
            art_clean["word_count"] = len(art_clean["text"].split())
            cleaned_articles.append(art_clean)
        else:
            dropped_count += 1

    save_jsonl(cleaned_articles, str(output_path))
    print(f"Filtered {dropped_count} articles failing length constraints.")
    print(f"[Person B - Step 1/2 Complete] Saved {len(cleaned_articles)} cleaned articles to {output_path}")


if __name__ == "__main__":
    main()
