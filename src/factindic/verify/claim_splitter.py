"""Atomic, typed claim splitting for summary sentences."""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Claim:
    claim_id: str
    text: str
    claim_type: str  # "number", "date", "entity", "relation"
    source_sentence: str


class ClaimSplitter:
    """
    Decomposes summary sentences into atomic propositions and tags each by type.
    """

    MONTHS = [
        "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
        "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
    ]

    DAYS = [
        "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ]

    def _determine_type(self, text: str) -> str:
        text_lower = text.lower()

        # 1. Date Check
        if any(d in text_lower for d in self.DAYS) or any(m in text_lower for m in self.MONTHS):
            return "date"
        if re.search(r"\b(19\d\d|20\d\d)\b", text):
            return "date"

        # 2. Number Check (digits, percentages, currency, counts)
        if re.search(r"\b\d+(?:[\.,]\d+)?%?\b", text) or any(w in text for w in ["करोड़", "लाख", "हजार", "crore", "lakh", "billion", "percent"]):
            return "number"

        # 3. Entity Check (Latin acronyms, uppercase names, institutions)
        if re.search(r"\b[A-Z][A-Za-z0-9\-]{1,}\b", text) or any(e in text for e in ["ISRO", "AIIMS", "CPCB", "RBI", "WHO", "CBSE", "HAL", "NHSRCL", "सरकार", "मंत्रालय", "आरबीआई", "इसरो"]):
            return "entity"

        # 4. Default: Relation
        return "relation"

    def split_claims(self, summary_text: str) -> List[Claim]:
        """
        Splits summary text into atomic claims based on punctuation and conjunctions,
        ensuring decimal numbers (e.g. 6.50% or 1.5) are preserved.
        """
        if not summary_text or not summary_text.strip():
            return []

        # Split into clauses by danda, newline, period NOT surrounded by digits, commas, and conjunctions
        raw_clauses = re.split(r"[।\n]|(?<!\d)\.(?!\d)|,\s*|\s+(?:तथा|और|एवं|\bbut\b|\band\b|\bwhile\b|\bwhereas\b)\s+", summary_text)

        claims: List[Claim] = []
        claim_counter = 1

        for clause in raw_clauses:
            if not clause:
                continue
            clause = clause.strip(" ,-–—।'\"")
            # Drop trivial fragments
            if len(clause.split()) < 3:
                continue

            ctype = self._determine_type(clause)
            claims.append(Claim(
                claim_id=f"c{claim_counter}",
                text=clause,
                claim_type=ctype,
                source_sentence=summary_text,
            ))
            claim_counter += 1

        # If splitting resulted in no claims, treat whole summary as single claim
        if not claims and summary_text.strip():
            claims.append(Claim(
                claim_id="c1",
                text=summary_text.strip(),
                claim_type=self._determine_type(summary_text),
                source_sentence=summary_text,
            ))

        return claims
