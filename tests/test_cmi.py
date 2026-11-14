"""Unit tests for CMI computation (Gambäck & Das 2014)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.mixing.cmi import compute_cmi


def test_pure_hindi_cmi():
    text = "केंद्रीय प्रदूषण नियंत्रण बोर्ड ने दिल्ली में वायु गुणवत्ता का सूचकांक जारी किया।"
    res = compute_cmi(text)
    assert res["cmi"] == 0.0
    assert res["w_hindi"] > 0
    assert res["w_english"] == 0


def test_pure_english_cmi():
    text = "The Central Pollution Control Board released the air quality index for Delhi."
    res = compute_cmi(text)
    assert res["cmi"] == 0.0
    assert res["w_hindi"] == 0
    assert res["w_english"] > 0


def test_code_mixed_cmi():
    text = "Delhi में pollution level लगातार increase हो रहा है।"
    res = compute_cmi(text)
    # Both Hindi and English present, so CMI must be > 0
    assert res["cmi"] > 0.0
    assert res["w_hindi"] > 0
    assert res["w_english"] > 0


def test_numbers_and_punctuation_as_neutral():
    text = "दिल्ली में 15 नवंबर 2024 को AQI 432 दर्ज किया गया।"
    res = compute_cmi(text)
    assert res["u_neutral"] > 0  # Digits and punctuation counted as neutral
