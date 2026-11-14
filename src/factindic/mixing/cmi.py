"""Gambäck & Das (2014) Code-Mixing Index (CMI) implementation."""

from typing import Dict, List, Tuple
from .script_utils import TokenScript, classify_script, tokenize_script_aware


def compute_cmi(text: str) -> Dict[str, float]:
    """
    Computes the Code-Mixing Index (CMI) according to Gambäck & Das (2014).

    Formula:
        CMI = 100 * (1 - (max(w1, w2) / (n - u))) if (n - u) > 0 else 0.0

    Where:
        w1: number of tokens in Language 1 (Hindi)
        w2: number of tokens in Language 2 (English)
        n: total token count
        u: language-independent tokens (numbers, punctuation, symbols)

    Returns:
        Dictionary containing:
        - cmi: float [0.0 to 100.0]
        - w_hindi: count of Hindi tokens
        - w_english: count of English tokens
        - u_neutral: count of language-independent tokens
        - total_tokens: total token count (n)
        - lang_dependent_tokens: (n - u)
        - mixing_fraction: fraction of minority language tokens
    """
    if not text or not text.strip():
        return {
            "cmi": 0.0,
            "w_hindi": 0,
            "w_english": 0,
            "u_neutral": 0,
            "total_tokens": 0,
            "lang_dependent_tokens": 0,
            "mixing_fraction": 0.0,
        }

    tokens = tokenize_script_aware(text)
    w_hi = 0
    w_en = 0
    u = 0

    for tok, script in tokens:
        if script == TokenScript.HINDI:
            w_hi += 1
        elif script == TokenScript.ENGLISH:
            w_en += 1
        elif script in (TokenScript.NUMBER, TokenScript.PUNCT, TokenScript.OTHER):
            u += 1

    n = len(tokens)
    lang_dep = n - u

    if lang_dep <= 0:
        cmi_score = 0.0
        minority_fraction = 0.0
    else:
        max_w = max(w_hi, w_en)
        cmi_score = 100.0 * (1.0 - (max_w / lang_dep))
        minority_w = min(w_hi, w_en)
        minority_fraction = minority_w / lang_dep

    return {
        "cmi": round(cmi_score, 2),
        "w_hindi": w_hi,
        "w_english": w_en,
        "u_neutral": u,
        "total_tokens": n,
        "lang_dependent_tokens": lang_dep,
        "mixing_fraction": round(minority_fraction, 4),
    }
