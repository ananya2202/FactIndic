"""Prepare and evaluate the FactIndic human annotation sample."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data/processed/human_annotation_sample.jsonl"
SHEET = ROOT / "results/tables/human_annotation_sheet.csv"
AGREEMENT = ROOT / "results/tables/annotator_agreement.csv"

LABELS = {"supported", "contradicted", "not_enough_info"}


def prepare():
    if SHEET.exists():
        existing = pd.read_csv(SHEET)
        if (
            "annotator_1" in existing.columns
            and "annotator_2" in existing.columns
            and existing[["annotator_1", "annotator_2"]].notna().any().any()
        ):
            raise FileExistsError(
                f"{SHEET} already contains annotations. "
                "Refusing to overwrite them."
            )

    df = pd.read_json(SAMPLE, lines=True)

    required = {
        "article_id", "version_id", "mixing_level",
        "claim_id", "claim_text", "verifier_verdict",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Sample is missing columns: {sorted(missing)}")

    df = df.rename(columns={
        "verifier_verdict": "automated_verifier_verdict"
    })

    df["annotator_1"] = ""
    df["annotator_2"] = ""

    columns = [
        "claim_id", "article_id", "version_id", "mixing_level",
        "claim_text", "automated_verifier_verdict",
        "annotator_1", "annotator_2",
    ]

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(SHEET, index=False)

    print("Annotation sheet created:", SHEET.relative_to(ROOT))
    print("Total claims:", len(df))
    print("\nClaims per tier:")
    print(df.groupby("mixing_level").size().to_string())
    print("\nAnnotator columns are intentionally blank.")
    print("Both annotators must label independently.")


def normalize(series):
    return series.astype(str).str.strip().str.lower().str.replace(
        " ", "_", regex=False
    )


def evaluate():
    if not SHEET.exists():
        raise FileNotFoundError(
            "Annotation sheet not found. Run with --prepare first."
        )

    df = pd.read_csv(SHEET, keep_default_na=False)

    required = {"mixing_level", "annotator_1", "annotator_2"}
    if not required.issubset(df.columns):
        raise ValueError(f"Annotation sheet must contain: {sorted(required)}")

    df["annotator_1"] = normalize(df["annotator_1"])
    df["annotator_2"] = normalize(df["annotator_2"])

    for column in ["annotator_1", "annotator_2"]:
        invalid = set(df[column]) - LABELS
        if invalid:
            raise ValueError(
                f"Invalid labels in {column}: {sorted(invalid)}. "
                f"Allowed labels: {sorted(LABELS)}"
            )

    incomplete = (
        (df["annotator_1"] == "") | (df["annotator_2"] == "")
    )

    if incomplete.any():
        raise ValueError(
            f"{int(incomplete.sum())} claims still need labels from both "
            "annotators. No agreement score has been calculated."
        )

    def calculate(group, tier):
        a1 = group["annotator_1"]
        a2 = group["annotator_2"]

        return {
            "mixing_level": tier,
            "sample_count": len(group),
            "observed_agreement": (a1 == a2).mean(),
            "cohen_kappa": cohen_kappa_score(a1, a2, labels=sorted(LABELS)),
        }

    results = [calculate(df, "overall")]

    for tier, group in df.groupby("mixing_level", sort=True):
        results.append(calculate(group, tier))

    output = pd.DataFrame(results)
    AGREEMENT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(AGREEMENT, index=False)

    print("\n===== ANNOTATOR AGREEMENT =====")
    print(output.round(4).to_string(index=False))
    print("\nSaved:", AGREEMENT.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    if args.prepare:
        prepare()
    else:
        evaluate()


if __name__ == "__main__":
    main()
