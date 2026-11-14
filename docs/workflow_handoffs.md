# FactIndic Team Workflow & Responsibilities (Team 8)

## 1. Project Personnel & Task Mapping

| Role | Person | Primary Responsibilities | Core Deliverables |
|---|---|---|---|
| **A** | **Ananya Chatterjee** | Verification & Metrics | Claim splitter, rule matchers, NLI verifier, `verdicts.jsonl`, agreement calculations |
| **B** | **Sidra Siddique** | Data & Mixed Versions | Sampling XL-Sum, cleaning, translation, 0/10/30/50/100% CMI generator, `versions.jsonl`, real Hinglish |
| **C** | **Vanshika** | LLM Runs & Revise Loop | Prompts (standard vs factual), cache manager, summary batches, revise loop, `summaries.jsonl` |
| **D** | **Ishika Singh** | Accuracy Checks & Assembly | ROUGE, BERTScore, human annotation sheets, spot checks, bootstrap intervals, publication plots & tables |

---

## 2. Handoff Protocol & Flow
```
[B: Data & Versions]
       │
       ▼  (versions.jsonl)
[C: LLM Runs & Summaries]
       │
       ▼  (summaries.jsonl)
[A: Verification Engine] ◄────────┐
       │                          │ (flagged claims & revisions,
       ▼  (verdicts.jsonl)        │  max 2 rounds)
[D: Checks, Analysis & Assembly] ─┘
```

### Spot-Check Policy
- Person D spot-checks 20 articles at every handoff before proceeding to the next phase.
- If data errors are spotted in translation/mixing, returned to Person B.
- If prompt formatting errors occur, returned to Person C.
- If claim verification produces false positives/negatives, verifier thresholds are retuned by Person A.
