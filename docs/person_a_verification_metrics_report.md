# Person A Technical Report: Verification Engine, NLI, and Factuality Metrics

**Author:** Person A (Ananya Chatterjee) — Team 8  
**Project:** FactIndic (CSET 346: Natural Language Processing)

---

## 1. Executive Summary
Person A is responsible for the core verification pipeline and factuality metric formulation of the FactIndic framework (Tasks 5, 8, 11, 14, 17, 18, and 20).
Key accomplishments include:
1. Setting up the cross-lingual NLI environment and benchmarking on IndicXNLI premise-hypothesis pairs.
2. Implementing an atomic, typed-claim splitter that categorizes claims into numbers, dates, entities, and relations while preserving decimal numbers.
3. Implementing deterministic rule matchers targeting numbers, dates, and named entities in both Devanagari and Latin scripts.
4. Implementing an NLI entailment engine supporting both local transformer models and high-precision semantic/morphological Indic heuristics.
5. Formulating and calculating summary-level **Hallucination Rate**, **Factual Recall**, and the **Key-Fact Weighted Score**.
6. Delivering `data/processed/verdicts.jsonl` to Person C (revision loop) and Person D (analysis).

---

## 2. Verification Architecture Overview

```
                      [Summary Text]
                            │
                            ▼
                [1. Atomic Claim Splitter]
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
       [2. Rule Matchers]     [3. NLI Engine]
     (Numbers, Dates, Names)   (Indic Entailment)
                 │                     │
                 └──────────┬──────────┘
                            ▼
            [4. Dual-Consensus Decision Engine]
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
            [Supported]          [Unsupported]
                                (Error Category)
```

A claim is declared **Supported** if and only if **both** rule-based verification passes AND the NLI engine predicts **Entailment** ($P_{\text{entail}} \ge \tau = 0.60$).
If a rule fails or NLI yields neutral/contradiction, the claim is flagged as **Unsupported** and categorized into a specific error typology.

---

## 3. Atomic Typed-Claim Splitter (`src/factindic/verify/claim_splitter.py`)
Whole-summary scores mask individual hallucinations. The splitter decomposes summary text into single-fact propositions using clause boundaries and coordinating conjunctions (`तथा`, `और`, `एवं`, `and`, `but`), while explicitly preserving decimals (e.g., `6.50%` or `1.5 meters`).

### Claim Typology (RQ4):
- **`number`**: Numeric quantities, currency, counts, percentages (`432`, `18%`, `22,500 करोड़`, `10 साल`).
- **`date`**: Temporal anchors, calendar days, months, years (`शुक्रवार`, `15 नवंबर`, `2024`, `October 2024`).
- **`entity`**: Acronyms, institutions, people, locations (`CPCB`, `ISRO`, `AIIMS`, `RBI`, `Delhi`, `Gujarat`).
- **`relation`**: General predicates, events, and subject-verb-object assertions.

---

## 4. Rule Matchers and Indic NLI (`src/factindic/verify/rules.py` & `nli.py`)

### A. Deterministic Rule Matching:
- **Number Matcher:** Extracts normalized digits, strips commas and percentage markers, and verifies that every number in the claim exists in the source article.
- **Entity Matcher:** Matches Latin acronyms and Hindi/English named entities against source occurrences.
- **Date Matcher:** Matches days of the week, months, and 4-digit years.

### B. Natural Language Inference (NLI):
- Evaluates premise (source article) vs hypothesis (claim).
- Computes probabilities for Entailment ($P_e$), Neutral ($P_n$), and Contradiction ($P_c$).
- Incorporates Indic morphological stem matching to handle Hindi inflections (e.g., `रखने` vs `रखी`, `पहुंचा` vs `पहुंच`).
- Polarity inversion detection: Flags contradictions when negation tokens (`नहीं`, `बिना`, `not`, `never`) differ between premise and claim.

---

## 5. Mathematical Formulations of Metrics

### A. Hallucination Rate:
$$\text{Hallucination Rate} = \frac{\text{Unsupported Claims}}{\text{Total Summary Claims}}$$

### B. Factuality Precision:
$$\text{Precision} = 1 - \text{Hallucination Rate} = \frac{\text{Supported Claims}}{\text{Total Summary Claims}}$$

### C. Factual Recall:
Key factual anchors ($K$) are extracted directly from the source article.
$$\text{Factual Recall} = \frac{\sum_{k \in K} w(k) \cdot \mathbb{I}(k \text{ covered by summary})}{\sum_{k \in K} w(k)}$$

### D. Key-Fact Weighted Score:
To ensure critical information (numbers, dates, entities) is prioritized over minor prose:
- Critical Facts ($w = 2.0$): Facts containing verified numbers, dates, or named entities.
- Standard Facts ($w = 1.0$): General assertions.
- Penalty ($\lambda = 1.0$): Penalty for critical claims that are hallucinated.

$$S_{\text{weighted}} = \max\left(0, \frac{\sum_{k} w(k) \cdot \mathbb{I}(k \text{ covered})}{\sum_{k} w(k)} - \lambda \cdot \frac{\text{Unsupported Critical Claims}}{\max(1, \text{Total Claims})}\right)$$

Sensitivity analysis is performed across critical weights $w \in \{1.0, 2.0, 3.0\}$.

---

## 6. Pilot Validation Results (`scripts/test_verifier_a.py`)
Running the diagnostic suite on pilot data demonstrated:
- **Indic NLI Test:** Correctly classified Direct Entailment (Hindi $P_e = 0.89$), Code-Mixed Entailment ($P_e = 0.90$), Contradiction ($P_c = 0.80$), and Extrapolation ($P_n = 0.75$).
- **Rule Verification:** Successfully caught 100% of injected number drifts (e.g. 22,500 drifted to 50,000) and entity swaps.
- **Summary Validation:** Achieved 0% hallucination on faithful summaries, and flagged 100% of simulated hallucinations on drifted summaries.

---

## 7. Artifacts and Handoff Delivery
- Splitter: `src/factindic/verify/claim_splitter.py`
- Rules: `src/factindic/verify/rules.py`
- NLI Verifier: `src/factindic/verify/nli.py`
- Combined Verifier: `src/factindic/verify/verifier.py`
- Verification Runner: `scripts/05_verify.py`
- Diagnostic Suite: `scripts/test_verifier_a.py`
- Handoff Output: `data/processed/verdicts.jsonl`
