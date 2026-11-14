"""Deterministic rule matchers for numbers, dates, and named entities."""

import re
from typing import Any, Dict, List, Set, Tuple


class RuleMatcher:
    """
    Verifies that numbers, dates, and named entities present in a summary claim
    are strictly grounded in the source text.
    """

    MONTHS = [
        "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
        "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    DAYS = [
        "सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ]

    KNOWN_ENTITIES = [
        "आरबीआई", "इसरो", "एम्स", "डब्ल्यूएचओ", "सीबीएसई", "एचएएल", "एनडीआरएफ", "नासा",
        "दिल्ली", "गुजरात", "महाराष्ट्र", "हिमाचल", "राजस्थान", "अहमदाबाद", "मुंबई",
        "RBI", "ISRO", "AIIMS", "WHO", "CBSE", "HAL", "NDRF", "NASA", "CPCB", "NHSRCL",
        "Delhi", "Gujarat", "Maharashtra", "Himachal", "Rajasthan", "Ahmedabad", "Mumbai"
    ]

    def _normalize_num(self, n_str: str) -> str:
        """Strips commas and percent signs for comparison."""
        return n_str.replace(",", "").rstrip("%").strip()

    def _extract_numbers(self, text: str) -> Set[str]:
        """Extracts normalized numeric strings from text."""
        raw = re.findall(r"\b\d+(?:[\.,]\d+)*%?", text)
        results = set()
        for r in raw:
            norm = self._normalize_num(r)
            if norm:
                results.add(norm)
                # Also include decimal rounded or integer part if decimal
                if "." in norm:
                    results.add(norm.split(".")[0])
        return results

    def _extract_entities(self, text: str) -> Set[str]:
        found = set()
        # Latin acronyms and capitalized entity names
        latin_ents = re.findall(r"\b[A-Z][A-Za-z0-9\-]{1,}\b", text)
        found.update(latin_ents)

        # Known Hindi and English geographic / institutional entities
        for ent in self.KNOWN_ENTITIES:
            if ent in text:
                found.add(ent)
        return found

    def _extract_dates(self, text: str) -> Set[str]:
        text_lower = text.lower()
        found = set()
        for d in self.DAYS:
            if d in text_lower:
                found.add(d)
        for m in self.MONTHS:
            if m in text_lower:
                found.add(m)
        years = re.findall(r"\b(19\d\d|20\d\d)\b", text)
        for y in years:
            found.add(y)
        return found

    def verify_claim(self, claim_text: str, source_text: str) -> Dict[str, Any]:
        """
        Runs rule-based verification on claim against source.
        Returns:
            passed: bool
            error_type: "none" | "number_drift" | "date_alteration" | "entity_swap"
            issues: list of issue descriptions
        """
        issues = []
        error_type = "none"

        # 1. Number Rule
        claim_nums = self._extract_numbers(claim_text)
        src_nums = self._extract_numbers(source_text)
        for num in claim_nums:
            if num not in src_nums:
                issues.append(f"Number '{num}' in claim is missing or altered from source.")
                error_type = "number_drift"

        # 2. Entity Rule
        claim_ents = self._extract_entities(claim_text)
        src_ents = self._extract_entities(source_text)
        for ent in claim_ents:
            if ent not in src_ents and ent.lower() not in source_text.lower():
                issues.append(f"Entity '{ent}' in claim is not found in source text.")
                if error_type == "none":
                    error_type = "entity_swap"

        # 3. Date Rule
        claim_dates = self._extract_dates(claim_text)
        src_dates = self._extract_dates(source_text)
        for dt in claim_dates:
            if dt not in src_dates and dt not in source_text.lower():
                issues.append(f"Temporal/date token '{dt}' in claim is not in source text.")
                if error_type == "none":
                    error_type = "date_alteration"

        passed = (len(issues) == 0)
        return {
            "passed": passed,
            "error_type": error_type,
            "issues": issues,
        }
