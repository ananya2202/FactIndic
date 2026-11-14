"""Verification engine: atomic claim splitting, rule-based matching, and NLI."""

from .claim_splitter import Claim, ClaimSplitter
from .key_facts import KeyFact, KeyFactExtractor
from .nli import NLIVerifier
from .rules import RuleMatcher
from .verifier import FactualityVerifier

__all__ = [
    "Claim",
    "ClaimSplitter",
    "KeyFact",
    "KeyFactExtractor",
    "RuleMatcher",
    "NLIVerifier",
    "FactualityVerifier",
]
