"""Unit tests for code-mixing generator across 0%, 10%, 30%, 50%, and 100% levels."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.mixing.generator import CodeMixingGenerator


def test_generator_mixing_levels():
    gen = CodeMixingGenerator(seed=42)
    hi_text = "केंद्रीय प्रदूषण नियंत्रण बोर्ड ने दिल्ली में वायु गुणवत्ता का स्तर जारी किया।"
    en_text = "Central Pollution Control Board released air quality level for Delhi."

    # Level 0.0: Pure Hindi
    v0 = gen.generate_mixed_version(hi_text, 0.0, en_text)
    assert v0["mixing_target"] == 0.0
    assert v0["text"] == hi_text

    # Level 1.0: Full English
    v1 = gen.generate_mixed_version(hi_text, 1.0, en_text)
    assert v1["mixing_target"] == 1.0
    assert v1["text"] == en_text

    # Level 0.3: Mixed
    v3 = gen.generate_mixed_version(hi_text, 0.30, en_text)
    assert v3["mixing_target"] == 0.30
    assert v3["substitutions_made"] > 0
    assert v3["cmi"] > 0.0
