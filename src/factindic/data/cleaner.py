"""Text cleaning, Unicode normalization, and quality filtering for FactIndic."""

import re
import unicodedata
from typing import Any, Dict, List


def clean_text(text: str) -> str:
    """
    Normalizes Unicode representation and cleans noisy artifacts in Hindi/English text.
    - NFKC Unicode normalization
    - Removes zero-width joiners and non-breaking spaces
    - Standardizes quotation marks and dashes
    - Normalizes multiple spaces into a single space
    """
    if not text:
        return ""

    # NFKC Normalization (unifies Devanagari combined characters and nuktas)
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width spaces / non-joiners that break tokenizers
    for zw_char in ["\u200b", "\u200c", "\u200d", "\ufeff", "\u00a0"]:
        text = text.replace(zw_char, " ")

    # Normalize quotes and dashes using explicit Unicode code points
    text = re.sub(r'[\u201c\u201d\u201e\u201f"«»]', '"', text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b'`]", "'", text)
    text = re.sub(r"[\u2013\u2014\u2212–—−]", "-", text)

    # Standardize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_words(text: str) -> int:
    """Counts whitespace-delimited tokens in cleaned text."""
    return len(clean_text(text).split())


def filter_article(
    article: Dict[str, Any],
    min_article_words: int = 50,
    max_article_words: int = 1500,
    min_summary_words: int = 8,
    max_summary_words: int = 200,
) -> bool:
    """Returns True if the article meets length and structural criteria."""
    text = article.get("text", "")
    summary = article.get("summary", "")

    if not text or not summary:
        return False

    n_article = count_words(text)
    n_summary = count_words(summary)

    if not (min_article_words <= n_article <= max_article_words):
        return False
    if not (min_summary_words <= n_summary <= max_summary_words):
        return False

    return True


def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Removes duplicate articles based on normalized title or text prefix."""
    seen = set()
    deduped = []
    for art in articles:
        key = clean_text(art.get("title", "")) or clean_text(art.get("text", ""))[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(art)
    return deduped
