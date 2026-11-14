"""Unit tests for text cleaning and Unicode normalization."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.cleaner import clean_text, filter_article


def test_unicode_normalization():
    # Combining characters and curly quotes
    noisy = "दिल्ली  में “प्रदूषण” का स्तर\u200b बढ़ रहा है।   "
    cleaned = clean_text(noisy)
    assert "\u200b" not in cleaned
    assert '“' not in cleaned
    assert '"प्रदूषण"' in cleaned
    assert "  " not in cleaned


def test_filter_article():
    art_valid = {
        "text": "यह एक लंबा समाचार आलेख है जिसमें पर्याप्त संख्या में शब्द उपस्थित हैं। " * 8,
        "summary": "यह एक संक्षिप्त सारांश है जिसमें आवश्यक विवरण हैं।"
    }
    assert filter_article(art_valid, min_article_words=10, min_summary_words=5)

    art_too_short = {"text": "बहुत छोटा", "summary": "छोटा"}
    assert not filter_article(art_too_short, min_article_words=20)
