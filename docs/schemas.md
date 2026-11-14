# Handoff file formats (agree on these first; change only by team decision)

All files are UTF-8, one JSON object per line (JSONL) unless stated. Keys are lowercase with underscores.
`article_id` is the XL-Sum id. `version` is the percent English: 0, 10, 30, 50 or 100.

## 1. versions.jsonl  (B -> C)
| field | meaning |
|---|---|
| article_id | XL-Sum article id |
| version | 0, 10, 30, 50, 100 |
| text | the article at this level |
| reference | the reference summary, transformed the same way |
| english_share | measured English share of language tokens (0 to 100) |
| cmi | Code-Mixing Index of `text` |
| numbers_ok | true if the set of numbers equals the original's |
| names_ok | true if names are unchanged |
| checks_passed | true if every script check passed |

## 2. summaries.jsonl  (C -> A)
| field | meaning |
|---|---|
| article_id, version | as above |
| prompt | "standard" or "factual" |
| round | 0 = first summary, 1 or 2 = revision rounds |
| summary | the model output |
| model, model_version | exact names, recorded for reproducibility |
| seconds | inference time for this call |
| created | ISO date-time |

## 3. key_facts.jsonl  (A, one set per original article)
| field | meaning |
|---|---|
| article_id, fact_id | ids |
| fact | one standalone sentence |
| type | number, date, entity, relation, other |
| weight | critical_weight for number/date/entity, else 1 |

## 4. verdicts.jsonl  (A -> D)
| field | meaning |
|---|---|
| article_id, version, prompt, round | identify the summary |
| claim_id, claim, type | one claim from the summary |
| nli_score | best entailment probability |
| evidence | best-matching source sentence |
| rule_issues | list of strings, empty if none |
| supported | true only if nli_score >= threshold and no rule issue |

## 5. labels.csv  (D, from the human check)
| column | meaning |
|---|---|
| claim_key | article_id:version:prompt:round:claim_id |
| annotator | a, b, c or d |
| label | supported / unsupported (optionally contradicted) |
| notes | free text |
The verifier's verdict is hidden from the label sheet.

## 6. metrics.csv  (D, results/tables/)
One row per article, version, prompt and round, with the metric columns. Plots and tables are built only from this file.

Example rows live next to each data folder as `*.example.jsonl` (these ARE committed).
