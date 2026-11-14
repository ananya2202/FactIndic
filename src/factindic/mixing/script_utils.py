"""Script detection and token classification for Devanagari and Latin text."""

import re
import string
from enum import Enum
from typing import List, Tuple


class TokenScript(str, Enum):
    HINDI = "hi"          # Devanagari script
    ENGLISH = "en"        # Latin script
    NUMBER = "num"        # Digits (Latin or Devanagari)
    PUNCT = "punct"       # Punctuation / symbols
    OTHER = "other"       # Mixed or unclassified


DEVANAGARI_RANGE = (0x0900, 0x097F)
DEVANAGARI_DIGITS = set("०१२३४५६७८९")


def is_devanagari_char(char: str) -> bool:
    """Checks if a character falls in the Devanagari Unicode block."""
    code = ord(char)
    return DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]


def classify_script(token: str) -> TokenScript:
    """
    Classifies a token into Hindi, English, Number, Punctuation, or Other.
    """
    if not token:
        return TokenScript.OTHER

    # Check for punctuation / symbols
    cleaned = token.strip(string.punctuation + "।'\"-–—()[]{}%:;,")
    if not cleaned:
        return TokenScript.PUNCT

    # Check for numbers / percentages
    if re.match(r"^[\d०१२३४५६७८९]+(?:\.[\d०१२३४५६७८९]+)?%?$", cleaned):
        return TokenScript.NUMBER

    has_deva = any(is_devanagari_char(c) for c in cleaned)
    has_latin = any("a" <= c.lower() <= "z" for c in cleaned)

    if has_deva and not has_latin:
        return TokenScript.HINDI
    elif has_latin and not has_deva:
        return TokenScript.ENGLISH
    else:
        # Mixed tokens or symbols
        return TokenScript.OTHER


def tokenize_script_aware(text: str) -> List[Tuple[str, TokenScript]]:
    """
    Tokenizes text while preserving word boundaries and assigning script tags.
    """
    # Regex splits words, numbers with decimals/percent, and punctuation separately
    tokens = re.findall(r"\b[\w\u0900-\u097F]+(?:[.\-][\w\u0900-\u097F]+)*%?|[^\w\s]", text)
    result = []
    for tok in tokens:
        script = classify_script(tok)
        result.append((tok, script))
    return result
