"""Synthetic code-mixing generator targeting controlled CMI levels."""

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .cmi import compute_cmi
from .script_utils import TokenScript, classify_script


class CodeMixingGenerator:
    """
    Generates controlled Hindi-English code-mixed text across target mixing levels:
    0% (pure Hindi), 10%, 30%, 50% mixed, and 100% (English).
    Combines intra-sentential loanword substitution and inter-sentential switching.
    """

    def __init__(self, lexicon_path: Optional[str] = None, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.lexicon = self._load_lexicon(lexicon_path)

    def _load_lexicon(self, lexicon_path: Optional[str]) -> Dict[str, str]:
        """Loads and flattens the Hindi-English parallel dictionary."""
        mapping: Dict[str, str] = {}
        if not lexicon_path:
            # Default location
            lexicon_path = str(Path(__file__).parent.parent.parent.parent / "data" / "dictionary" / "hi_en_lexicon.json")

        path = Path(lexicon_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for category, entries in data.items():
                    for hi_word, en_word in entries.items():
                        mapping[hi_word] = en_word
        return mapping

    def _split_sentences(self, text: str) -> List[str]:
        """Splits Hindi or English text into sentences handling both '.' and '।'."""
        return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", str(text)) if len(s.strip()) > 3]

    def _substitute_words(self, text: str, ratio: float, rng: random.Random) -> Tuple[str, int]:
        """Substitutes eligible Hindi tokens with English loanwords."""
        tokens = re.findall(r"[\w\u0900-\u097F]+|[^\w\s]|\s+", text)
        eligible = [i for i, tok in enumerate(tokens) if tok.strip("।,?!-–—\"'") in self.lexicon]
        if not eligible:
            return text, 0
        k = max(1, int(len(eligible) * ratio))
        chosen = set(rng.sample(eligible, min(k, len(eligible))))
        out = [self.lexicon[tokens[i].strip("।,?!-–—\"'")] if i in chosen else tokens[i] for i in range(len(tokens))]
        return "".join(out), len(chosen)

    def generate_mixed_version(
        self,
        hindi_text: str,
        target_ratio: float,
        english_text: Optional[str] = None,
        seed_offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Generates a version of text at the specified mixing target.
        - target_ratio == 0.0: Pure Hindi (0% English)
        - target_ratio == 1.0: Full English translation
        - target_ratio in (0.0, 1.0): Synthetically mixed text targeting CMI ~ target_ratio * 100
        """
        rng = random.Random(self.seed + seed_offset + int(target_ratio * 1000))

        if target_ratio <= 0.001:
            cmi_info = compute_cmi(hindi_text)
            return {
                "text": hindi_text,
                "mixing_target": 0.0,
                "cmi": cmi_info["cmi"],
                "token_counts": {
                    "hindi": cmi_info["w_hindi"],
                    "english": cmi_info["w_english"],
                    "other": cmi_info["u_neutral"],
                },
                "substitutions_made": 0,
            }

        if target_ratio >= 0.999:
            text_out = english_text if english_text else hindi_text
            cmi_info = compute_cmi(text_out)
            return {
                "text": text_out,
                "mixing_target": 1.0,
                "cmi": cmi_info["cmi"],
                "token_counts": {
                    "hindi": cmi_info["w_hindi"],
                    "english": cmi_info["w_english"],
                    "other": cmi_info["u_neutral"],
                },
                "substitutions_made": 0,
            }

        # For intermediate mixing targets (0.10, 0.30, 0.50):
        hi_sents = self._split_sentences(hindi_text)
        en_sents = self._split_sentences(english_text) if english_text else []

        if target_ratio <= 0.12:
            sent_replace_ratio = 0.06
            word_ratio = 0.50
        elif target_ratio <= 0.32:
            sent_replace_ratio = 0.38
            word_ratio = 0.50
        else:
            sent_replace_ratio = 0.60
            word_ratio = 0.50

        n_sents = len(hi_sents)
        n_replace = int(round(n_sents * sent_replace_ratio))
        replace_indices = set(rng.sample(range(n_sents), min(n_replace, len(en_sents)))) if n_replace > 0 and len(en_sents) > 0 else set()

        mixed_sents = []
        subs_made = len(replace_indices)
        for i, s in enumerate(hi_sents):
            if i in replace_indices and i < len(en_sents):
                mixed_sents.append(en_sents[i])
            else:
                subbed_s, w_subs = self._substitute_words(s, word_ratio, rng)
                mixed_sents.append(subbed_s)
                subs_made += w_subs

        mixed_text = " ".join(mixed_sents) if mixed_sents else hindi_text
        cmi_info = compute_cmi(mixed_text)

        return {
            "text": mixed_text,
            "mixing_target": target_ratio,
            "cmi": cmi_info["cmi"],
            "token_counts": {
                "hindi": cmi_info["w_hindi"],
                "english": cmi_info["w_english"],
                "other": cmi_info["u_neutral"],
            },
            "substitutions_made": subs_made,
        }

    def generate_all_versions(
        self,
        article_id: str,
        hindi_text: str,
        reference_summary: str,
        english_text: str,
        english_summary: str,
        mixing_levels: List[float] = [0.0, 0.10, 0.30, 0.50, 1.0],
    ) -> List[Dict[str, Any]]:
        """Generates all 5 experimental versions for an article and reference."""
        versions = []
        labels = {
            0.0: "Hindi (0% English)",
            0.10: "10% English mixed",
            0.30: "30% English mixed",
            0.50: "50% English mixed",
            1.0: "English (full)",
        }

        seed_offset = abs(hash(article_id)) % 10000

        for level in mixing_levels:
            version_id = f"{article_id}_mix{level:.2f}"
            article_mix = self.generate_mixed_version(
                hindi_text=hindi_text,
                target_ratio=level,
                english_text=english_text,
                seed_offset=seed_offset,
            )

            # Apply identical transformation to reference summary
            summary_mix = self.generate_mixed_version(
                hindi_text=reference_summary,
                target_ratio=level,
                english_text=english_summary,
                seed_offset=seed_offset + 50,
            )

            versions.append({
                "article_id": article_id,
                "version_id": version_id,
                "mixing_target": level,
                "language_label": labels.get(level, f"{int(level*100)}% mixed"),
                "cmi": article_mix["cmi"],
                "text": article_mix["text"],
                "reference_summary": summary_mix["text"],
                "ref_cmi": summary_mix["cmi"],
                "substitutions": article_mix["substitutions_made"],
                "word_count": len(article_mix["text"].split()),
                "token_counts": article_mix["token_counts"],
            })

        return versions
