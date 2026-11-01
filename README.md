# FactIndic

Factuality-aware summarization for Hindi-English code-mixed text (CSET 346, NLP).

**Question.** Does mixing English into Hindi make LLM summaries less factually consistent, and can factuality-aware
prompting and a verify-and-revise loop reduce that loss?

## Layout
```
configs/        config.yaml (every setting)
data/           local data, not in Git (see CONTRIBUTING.md); example rows only
docs/           schemas.md (handoff formats), labeling_guide.md
src/factindic/  data, mixing, llm, verify, metrics, analysis
scripts/        numbered entry points, one per pipeline step
app/            Streamlit demo (paste text, see summary and factuality)
results/        tables/ and figures/ (committed)
paper/ slides/  deliverables
```

## Pipeline (scripts run in order)
| Script | Step | Owner |
|---|---|---|
| 01_sample_clean.py | sample 250 XL-Sum Hindi test articles, clean | B |
| 02_translate.py | English translation of each article | B |
| 03_make_versions.py | 10/30/50% mixed versions + checks | B |
| 04_summarize.py | standard vs factuality-aware summaries | C |
| 05_verify.py | claims, NLI + rules, key facts, scores | A |
| 06_revise.py | revise loop, max 2 rounds (with 05) | C + A |
| 07_metrics.py | ROUGE, BERTScore, length, time | D |
| 08_human_labels.py | build label sheet, compute kappa | D |
| 09_real_hinglish.py | MUTANT and GupShup side check | B, C, A, D |
| 10_analysis.py | plots, bootstrap intervals, tables | D |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # add your keys
```

## Status
Scripts are stubs: each states its inputs, outputs and owner. See `docs/schemas.md` for file formats.
