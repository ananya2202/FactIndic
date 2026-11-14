# Person B Technical Report: Data Preparation, Translation, and Controlled Code-Mixing

**Author:** Person B (Sidra Siddique) — Team 8  
**Project:** FactIndic (CSET 346: Natural Language Processing)

---

## 1. Executive Summary
Person B is responsible for the upstream data generation and quality assurance of the FactIndic pipeline (Tasks 1, 2, 3, 4, 6, 12, and 20). This includes:
1. Sampling and cleaning BBC Hindi news articles from the XL-Sum benchmark.
2. Generating aligned English translations with automated numerical and entity integrity verification.
3. Constructing a parameterized synthetic Hindi-English code-mixing generator targeting 0%, 10%, 30%, 50%, and 100% mixing levels.
4. Measuring and validating Gambäck & Das (2014) Code-Mixing Index (CMI) against empirical Hinglish datasets (MUTANT and PHINC).
5. Producing the formal handoff file `data/processed/versions.jsonl` consumed by Persons C and A.

---

## 2. Dataset Sampling and Cleaning Pipeline (`scripts/01_sample_clean.py`)
Articles are sampled from the test split of the BBC XL-Sum Hindi dataset.

### Cleaning and Normalization Steps:
- **NFKC Unicode Normalization:** Unifies pre-composed Devanagari characters, matras, and nuktas (e.g. `क़`, `फ़`, `ज़`).
- **Zero-Width Artifact Stripping:** Eliminates hidden Unicode formatting tokens (`\u200b`, `\u200c`, `\u200d`, `\ufeff`) that corrupt subword tokenizers.
- **Punctuation & Quote Standardization:** Normalizes smart quotes (`“”‘’`) and typographical dashes (`—–−`).
- **Length Filtering:** Constrains article word counts ($50 \le N_{\text{article}} \le 1500$) and summary lengths ($8 \le N_{\text{summary}} \le 200$).
- **Deduplication:** Hashed normalization across title and text prefixes to eliminate redundant wire stories.

---

## 3. English Translation and Alignment Verification (`scripts/02_translate.py`)
To enable cross-lingual evaluation (RQ1: 100% English control baseline), each cleaned Hindi article and reference summary is paired with an English translation.

### Quality Assurance Protocol:
Machine translation can unintentionally drop numerical facts or distort names. We enforce an automated alignment checker `verify_translation_alignment`:
- Extracts numbers, decimals, and percentages from the Hindi source text.
- Extracts Latin acronyms and named entities.
- Flags any discrepancy where a source numerical fact is absent in the target translation.
- Articles with unresolved alignment errors are re-verified prior to handoff.

---

## 4. Controlled Code-Mixing Engine (`scripts/03_make_versions.py`)
To measure the impact of mixing on factual consistency (RQ1), we generate 5 controlled versions of each article:
- **0% (Pure Hindi):** Original Devanagari text.
- **10% Mixed:** Contextual loanword replacement targeting low mixing.
- **30% Mixed:** Moderate mixing representing standard Hinglish news articles.
- **50% Mixed:** High mixing representing conversational Hinglish.
- **100% (English):** Full English translation.

### Gambäck & Das (2014) Code-Mixing Index (CMI):
$$CMI = \begin{cases} 100 \times \left(1 - \frac{\max(w_1, w_2)}{n - u}\right) & \text{if } n > u \\ 0 & \text{otherwise} \end{cases}$$
where:
- $w_1$: Hindi tokens (Devanagari script)
- $w_2$: English tokens (Latin script)
- $n$: Total tokens
- $u$: Language-independent tokens (digits, punctuation, symbols)

### Comparison with External Literature Benchmarks:
| Target Level | Mean CMI Achieved | Empirical Benchmark Reference |
|---|---|---|
| **0% (Hindi)** | 1.87% | Monolingual baseline |
| **10% Mixed** | 11.01% | Incidental code-mixing |
| **30% Mixed** | 19.20% – 32.86% | **MUTANT news benchmark (CMI: 28.5 – 34.0)** & MedSumm (CMI: 30.5) |
| **50% Mixed** | 19.20% – 32.86% | **PHINC social media benchmark (CMI: 32.0 – 41.0)** |
| **100% (English)** | 0.00% | Monolingual control |

### Reference Summary Transformation:
To ensure fair ROUGE and BERTScore evaluation (Person D), the reference summaries undergo the exact same linguistic transformation as their source articles.

---

## 5. Artifacts and Handoff Delivery
- Cleaned Articles: `data/processed/clean_articles.jsonl`
- Translations: `data/processed/translations.jsonl`
- Final Multi-Version Corpus: `data/processed/versions.jsonl`
- Automated Lexicon: `data/dictionary/hi_en_lexicon.json`
