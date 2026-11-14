"""Extraction and weighting of key factual assertions from source articles."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class KeyFact:
    fact_id: str
    text: str
    fact_type: str  # "number", "date", "entity", "standard"
    weight: float    # 2.0 for critical (number/date/entity), 1.0 for standard


class KeyFactExtractor:
    """
    Extracts key factual anchors directly from the source article.
    """

    def __init__(self, critical_weight: float = 2.0, standard_weight: float = 1.0):
        self.critical_weight = critical_weight
        self.standard_weight = standard_weight

    def extract_key_facts(self, article_text: str) -> List[KeyFact]:
        """
        Parses salient factual propositions from the article.
        """
        facts: List[KeyFact] = []
        sentences = [s.strip() for s in re.split(r"[।\.\n]", article_text) if s.strip()]

        fact_idx = 1
        for sent in sentences[:6]:  # Target lead paragraphs where core news facts reside
            # Find numbers and percentages
            numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", sent)
            # Find Latin acronyms / entities
            entities = re.findall(r"\b[A-Z][A-Za-z0-9\-]{2,}\b", sent)

            if numbers:
                facts.append(KeyFact(
                    fact_id=f"kf_{fact_idx}",
                    text=f"Numeric assertion in: {sent[:80]}",
                    fact_type="number",
                    weight=self.critical_weight,
                ))
                fact_idx += 1
            elif entities:
                facts.append(KeyFact(
                    fact_id=f"kf_{fact_idx}",
                    text=f"Entity assertion regarding: {', '.join(entities)}",
                    fact_type="entity",
                    weight=self.critical_weight,
                ))
                fact_idx += 1
            else:
                facts.append(KeyFact(
                    fact_id=f"kf_{fact_idx}",
                    text=sent[:80],
                    fact_type="standard",
                    weight=self.standard_weight,
                ))
                fact_idx += 1

        return facts
