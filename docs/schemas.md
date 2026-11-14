# FactIndic Schema Specifications

This document defines the schema for every inter-person handoff file in the FactIndic pipeline.
All files follow the **JSON Lines (`.jsonl`)** format where each line is a valid JSON object.

---

## 1. `versions.jsonl`
**Produced by:** Person B (Data & Versions) — Step 4  
**Consumed by:** Person C (LLM Runs) — Step 5  
**Path:** `data/processed/versions.jsonl`

Each article generates 5 version rows corresponding to mixing levels `[0.0, 0.1, 0.3, 0.5, 1.0]`.

```json
{
  "article_id": "xlsum_hi_001",
  "version_id": "xlsum_hi_001_mix0.30",
  "mixing_target": 0.30,
  "cmi": 28.45,
  "language_label": "30% mixed",
  "text": "भारत में pollution का level लगातार increase हो रहा है...",
  "reference_summary": "Pollution levels in India are continuously rising...",
  "word_count": 284,
  "token_counts": {
    "hindi": 182,
    "english": 78,
    "other": 24
  },
  "protected_elements": {
    "numbers": ["2024", "15%"],
    "entities": ["Delhi", "CPCB"]
  }
}
```

---

## 2. `summaries.jsonl`
**Produced by:** Person C (LLM Runs) — Step 5  
**Consumed by:** Person A (Verification) — Step 6–8  
**Path:** `data/processed/summaries.jsonl`

```json
{
  "article_id": "xlsum_hi_001",
  "version_id": "xlsum_hi_001_mix0.30",
  "prompt_type": "factuality_aware",
  "round": 0,
  "summary": "भारत में pollution level 15% increase हुआ है, जैसा CPCB reports confirm करती हैं।",
  "model": "gemini-1.5-flash",
  "latency_ms": 780,
  "word_count": 14,
  "prompt_hash": "a1b2c3d4...",
  "timestamp": "2026-11-14T01:00:00Z"
}
```

---

## 3. `verdicts.jsonl`
**Produced by:** Person A (Verification & Metrics) — Step 6–8  
**Consumed by:** Person C (Revision Loop) and Person D (Accuracy & Analysis)  
**Path:** `data/processed/verdicts.jsonl`

```json
{
  "article_id": "xlsum_hi_001",
  "version_id": "xlsum_hi_001_mix0.30",
  "prompt_type": "factuality_aware",
  "round": 0,
  "claims": [
    {
      "claim_id": "c1",
      "text": "pollution level 15% increase हुआ है",
      "type": "number",
      "rule_passed": true,
      "nli_label": "entailment",
      "nli_score": 0.94,
      "verdict": "supported"
    },
    {
      "claim_id": "c2",
      "text": "CPCB reports confirm करती हैं",
      "type": "entity",
      "rule_passed": true,
      "nli_label": "entailment",
      "nli_score": 0.89,
      "verdict": "supported"
    }
  ],
  "key_facts_evaluated": [
    {
      "fact_id": "kf1",
      "text": "Pollution level rose by 15%",
      "weight": 2.0,
      "covered": true
    }
  ],
  "summary_metrics": {
    "total_claims": 2,
    "supported_claims": 2,
    "hallucinated_claims": 0,
    "hallucination_rate": 0.0,
    "factual_recall": 1.0,
    "weighted_score": 2.0
  }
}
```

---

## 4. `revised_summaries.jsonl`
**Produced by:** Person C + Person A — Step 9  
**Path:** `data/processed/revised_summaries.jsonl`

```json
{
  "article_id": "xlsum_hi_001",
  "version_id": "xlsum_hi_001_mix0.30",
  "prompt_type": "standard",
  "round": 1,
  "original_summary": "...",
  "flagged_unsupported_claims": ["claim text..."],
  "revised_summary": "...",
  "re_verified": true,
  "hallucination_rate_before": 0.33,
  "hallucination_rate_after": 0.0,
  "status": "resolved"
}
```

---

## 5. `metrics.csv` / `metrics.jsonl`
**Produced by:** Person D (Assembly & Standard Metrics) — Step 10  
**Path:** `results/tables/metrics_summary.csv`

Columns:
`article_id`, `version_id`, `mixing_level`, `prompt_type`, `round`, `rouge1_f1`, `rouge2_f1`, `rougeL_f1`, `bert_score_f1`, `hallucination_rate`, `factual_recall`, `weighted_score`, `compression_ratio`, `summary_words`, `latency_ms`.

---

## 6. `human_labels.jsonl`
**Produced by:** Person D (Annotation Sheet) / Human Readers — Step 11  
**Path:** `data/processed/human_labels.jsonl`

```json
{
  "claim_id": "xlsum_hi_001_mix0.30_c1",
  "article_id": "xlsum_hi_001",
  "version_id": "xlsum_hi_001_mix0.30",
  "claim_text": "pollution level 15% increase हुआ है",
  "annotator_id": "annotator_1",
  "human_verdict": "supported",
  "verifier_verdict": "supported",
  "agreement": true,
  "error_type": "none",
  "notes": "Verified against source article paragraph 2"
}
```
