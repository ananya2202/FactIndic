"""NLI entailment verifier for Hindi and code-mixed premise-hypothesis pairs."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class NLIVerifier:
    """
    Evaluates Natural Language Inference (NLI) between source context (premise)
    and summary claim (hypothesis).
    Outputs probabilities for Entailment, Neutral, and Contradiction.
    """

    HINDI_STOPWORDS: Set[str] = {
        "का", "की", "के", "में", "से", "को", "ने", "पर", "है", "हैं", "था", "थी", "थे", "हो",
        "तथा", "और", "एवं", "व", "या", "भी", "ही", "तो", "अब", "तक",
        "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "are", "was", "were"
    }

    def __init__(self, entailment_threshold: float = 0.60, contradiction_threshold: float = 0.35):
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self._hf_pipeline = None
        self._init_backend()

    def _init_backend(self) -> None:
        """Attempts to load local HuggingFace NLI model if installed; otherwise uses semantic fallback."""
        try:
            from transformers import pipeline
            self._hf_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=-1)
        except Exception:
            self._hf_pipeline = None

    def _stem_indic(self, word: str) -> str:
        """Lightweight Indic and English stemmer for morphological suffix matching."""
        word = word.lower().strip("।,?!-–—\"'")
        # Strip common Hindi verbal/inflectional suffixes
        suffixes = ["ियों", "ियां", "यों", "ओं", "कर", "ते", "ती", "ता", "ने", "ना", "या", "यी", "ये", "ा", "ी", "े"]
        for suf in suffixes:
            if len(word) > len(suf) + 2 and word.endswith(suf):
                return word[:-len(suf)]
        return word

    def _semantic_overlap_score(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        High-precision semantic entailment heuristic supporting Hindi, English, and code-mixed pairs.
        Analyzes lexical grounding, morphological stems, token containment, and negation polarity.
        """
        prem_raw_tokens = re.findall(r"[\w\u0900-\u097F]+(?:\.[\w\u0900-\u097F]+)?%?", premise.lower())
        hyp_raw_tokens = re.findall(r"[\w\u0900-\u097F]+(?:\.[\w\u0900-\u097F]+)?%?", hypothesis.lower())

        if not hyp_raw_tokens:
            return 0.1, 0.8, 0.1

        # Check for negation mismatch
        negations = {"नहीं", "ना", "न", "not", "no", "never", "without", "बिना"}
        hyp_has_neg = any(w in negations for w in hyp_raw_tokens)
        prem_has_neg = any(w in negations for w in prem_raw_tokens)

        # Significant polarity flip -> contradiction
        if hyp_has_neg != prem_has_neg and any(w in prem_raw_tokens for w in hyp_raw_tokens if w not in negations):
            return 0.05, 0.15, 0.80

        # Filter content tokens
        content_words = [w for w in hyp_raw_tokens if w not in self.HINDI_STOPWORDS and len(w) >= 1]
        if not content_words:
            return 0.70, 0.25, 0.05

        prem_stems = {self._stem_indic(w) for w in prem_raw_tokens}
        prem_words = set(prem_raw_tokens)

        matched = 0
        for w in content_words:
            stem = self._stem_indic(w)
            # Match exact or stem
            if w in prem_words or stem in prem_stems or any(stem in p or p in stem for p in prem_stems if len(stem) >= 3):
                matched += 1

        coverage = matched / len(content_words)

        if coverage >= 0.50:
            p_entail = min(0.96, 0.65 + (coverage * 0.30))
            p_contra = 0.04
            p_neut = max(0.0, 1.0 - p_entail - p_contra)
        elif coverage >= 0.30:
            p_neut = 0.55
            p_entail = 0.35
            p_contra = 0.10
        else:
            p_neut = 0.75
            p_contra = 0.20
            p_entail = 0.05

        return round(p_entail, 3), round(p_neut, 3), round(p_contra, 3)

    def score(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Scores NLI pair and determines classification label.
        """
        if self._hf_pipeline is not None:
            try:
                out = self._hf_pipeline(f"{premise} </s></s> {hypothesis}")
                label = out[0]["label"].lower()
                score = out[0]["score"]
                return {
                    "label": label,
                    "entailment_score": score if "entail" in label else 0.1,
                    "neutral_score": score if "neut" in label else 0.1,
                    "contradiction_score": score if "contra" in label else 0.1,
                }
            except Exception:
                pass

        p_ent, p_neut, p_cont = self._semantic_overlap_score(premise, hypothesis)

        if p_cont >= self.contradiction_threshold:
            verdict = "contradiction"
        elif p_ent >= self.entailment_threshold:
            verdict = "entailment"
        else:
            verdict = "neutral"

        return {
            "label": verdict,
            "entailment_score": p_ent,
            "neutral_score": p_neut,
            "contradiction_score": p_cont,
        }
