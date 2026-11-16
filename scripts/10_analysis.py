"""Generate FactIndic analysis tables, bootstrap CIs, and figures."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
DATA = ROOT / "data" / "processed"

B = 1000
SEED = 42
TIERS = [0.0, 0.1, 0.3, 0.5, 1.0]
METRICS = [
    "hallucination_rate",
    "factual_recall",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
]


def load_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def bootstrap_confidence_intervals(df):
    """Article-cluster bootstrap, preserving repeated prompt observations."""
    rng = np.random.default_rng(SEED)
    output = []

    for tier in TIERS:
        tier_df = df[df["mixing_level"].astype(float) == tier]

        for metric in METRICS:
            valid = tier_df[["article_id", metric]].copy()
            valid[metric] = pd.to_numeric(valid[metric], errors="coerce")
            valid = valid.dropna()

            if valid.empty:
                continue

            # Average within article first, so two prompt observations
            # from one article are treated as a single sampling cluster.
            article_means = (
                valid.groupby("article_id")[metric].mean().to_numpy()
            )

            estimates = np.empty(B)
            for i in range(B):
                sample = rng.choice(
                    article_means,
                    size=len(article_means),
                    replace=True,
                )
                estimates[i] = sample.mean()

            output.append({
                "mixing_level": tier,
                "metric": metric,
                "estimate": float(article_means.mean()),
                "ci_95_lower": float(np.percentile(estimates, 2.5)),
                "ci_95_upper": float(np.percentile(estimates, 97.5)),
                "bootstrap_samples": B,
                "resampling_unit": "article_id",
            })

    result = pd.DataFrame(output)
    result.to_csv(TABLES / "bootstrap_confidence_intervals.csv", index=False)
    return result


def create_revision_comparison():
    rows = load_jsonl(DATA / "revised_summaries.jsonl")
    df = pd.DataFrame(rows)

    for col in ["hallucination_rate_before", "hallucination_rate_after"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["hallucination_change"] = (
        df["hallucination_rate_after"] - df["hallucination_rate_before"]
    )

    grouped = (
        df.groupby(["mixing_level", "prompt_type"], dropna=False)
        .agg(
            sample_count=("article_id", "count"),
            mean_hallucination_before=("hallucination_rate_before", "mean"),
            mean_hallucination_after=("hallucination_rate_after", "mean"),
            mean_hallucination_change=("hallucination_change", "mean"),
            mean_rounds_executed=("rounds_executed", "mean"),
            resolved_count=("status", lambda x: (x == "resolved").sum()),
            improved_count=("status", lambda x: (x == "improved").sum()),
            unchanged_count=("status", lambda x: (x == "unchanged").sum()),
        )
        .reset_index()
        .sort_values(["mixing_level", "prompt_type"])
    )

    grouped.to_csv(TABLES / "revise_loop_comparison.csv", index=False)

    # Plot unsupported claims across revision rounds for the same summaries.
    trace_path = DATA / "revision_round_trace_eval.jsonl"
    trace_rows = load_jsonl(trace_path)

    round_data = {}
    for row in trace_rows:
        key = (row["article_id"], row["version_id"])
        round_data.setdefault(key, {})[int(row["round"])] = int(
            row["unsupported_claims"]
        )

    matched = {
        key: rounds
        for key, rounds in round_data.items()
        if all(r in rounds for r in (0, 1, 2))
    }

    if not matched:
        raise ValueError("No summaries found in all three revision rounds.")

    round_numbers = [0, 1, 2]
    unsupported_totals = [
        sum(rounds[r] for rounds in matched.values())
        for r in round_numbers
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        round_numbers,
        unsupported_totals,
        marker="o",
        linewidth=2,
        label="Unsupported claims",
    )
    for x, count in zip(round_numbers, unsupported_totals):
        plt.annotate(
            f"{count:,}",
            (x, count),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.xticks(round_numbers, ["Round 0", "Round 1", "Round 2"])
    plt.xlabel("Revision round")
    plt.ylabel("Total unsupported claims")
    plt.title(
        "Unsupported Claims Across Revision Rounds"
    )
    plt.suptitle(
        f"Same {len(matched)} summaries compared in every round",
        fontsize=9,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "revise_loop_progression.png", dpi=200)
    plt.close()

    return grouped


def create_error_breakdown():
    rows = load_jsonl(DATA / "verdicts.jsonl")

    error_types = [
        "date_alteration",
        "entity_swap",
        "number_drift",
        "relation_distortion",
        "unsupported_claim",
    ]

    counts = {
        (tier, error_type): 0
        for tier in TIERS
        for error_type in error_types
    }

    for row in rows:
        tier = float(row["mixing_level"])
        distribution = row.get("error_distribution") or {}

        for error_type in error_types:
            count = int(distribution.get(error_type, 0) or 0)
            counts[(tier, error_type)] += count

    output = []

    for tier in TIERS:
        # Exclude claims labelled "none": only actual errors
        # contribute to the denominator.
        tier_total = sum(
            counts[(tier, error_type)]
            for error_type in error_types
        )

        for error_type in error_types:
            count = counts[(tier, error_type)]

            output.append({
                "mixing_level": tier,
                "error_type": error_type,
                "count": count,
                "proportion_within_tier": (
                    count / tier_total if tier_total else 0
                ),
            })

    result = pd.DataFrame(output)
    result.to_csv(TABLES / "error_type_breakdown.csv", index=False)
    return result


def create_prompt_comparison(df):
    grouped = (
        df.groupby(["mixing_level", "prompt_type"])
        .agg(
            sample_count=("article_id", "count"),
            hallucination_rate=("hallucination_rate", "mean"),
            factual_recall=("factual_recall", "mean"),
            rouge1_f1=("rouge1_f1", "mean"),
            rouge2_f1=("rouge2_f1", "mean"),
            rougeL_f1=("rougeL_f1", "mean"),
        )
        .reset_index()
    )

    tiers = TIERS
    prompts = sorted(grouped["prompt_type"].dropna().unique())
    x = np.arange(len(tiers))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for i, prompt in enumerate(prompts):
        part = (
            grouped[grouped["prompt_type"] == prompt]
            .set_index("mixing_level")
            .reindex(tiers)
        )
        offset = (i - (len(prompts) - 1) / 2) * width
        label = prompt.replace("_", " ").title()

        axes[0].bar(
            x + offset, part["hallucination_rate"],
            width, label=label,
        )
        axes[1].bar(
            x + offset, part["rougeL_f1"],
            width, label=label,
        )

    axes[0].set_ylabel("Mean hallucination rate")
    axes[0].set_title("Hallucination Rate by Prompt Type")
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].set_ylabel("Mean ROUGE-L F1")
    axes[1].set_title("ROUGE-L by Prompt Type")
    axes[1].set_xlabel("Mixing tier")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{int(t * 100)}%" for t in tiers])
    axes[1].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURES / "prompt_comparison.png", dpi=200)
    plt.close(fig)

    return grouped


def create_hallucination_curve(df):
    summary = (
        df.groupby("mixing_level")["hallucination_rate"]
        .agg(["mean", "count", "std"])
        .reindex(TIERS)
    )

    plt.figure(figsize=(9, 5))
    plt.plot(
        summary.index,
        summary["mean"],
        marker="o",
        linewidth=2,
    )
    plt.xticks(TIERS, [f"{int(t * 100)}%" for t in TIERS])
    plt.xlabel("Mixing tier")
    plt.ylabel("Mean hallucination rate")
    plt.title("Hallucination Rate Across Mixing Tiers")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "mixing_curve_hallucination.png", dpi=200)
    plt.close()


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics_path = TABLES / "metrics_summary.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "metrics_summary.csv is missing. Run scripts/07_metrics.py first."
        )

    df = pd.read_csv(metrics_path)

    required = {"article_id", "mixing_level", "prompt_type", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing columns: {sorted(missing)}")

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    print("Generating article-cluster bootstrap confidence intervals...")
    ci = bootstrap_confidence_intervals(df)

    print("Generating revision comparison...")
    revision = create_revision_comparison()

    print("Generating error type breakdown...")
    errors = create_error_breakdown()

    print("Generating prompt comparison...")
    prompt = create_prompt_comparison(df)

    print("Generating mixing curve...")
    create_hallucination_curve(df)

    print("\n===== BOOTSTRAP CONFIDENCE INTERVALS =====")
    print(ci.round(4).to_string(index=False))

    print("\n===== REVISION COMPARISON =====")
    print(revision.round(4).to_string(index=False))

    print("\n===== ERROR TYPE BREAKDOWN =====")
    print(errors.to_string(index=False))

    print("\n===== PROMPT COMPARISON =====")
    print(prompt.round(4).to_string(index=False))

    print("\nSaved analysis outputs:")
    for path in sorted(TABLES.glob("*.csv")):
        print(" ", path.relative_to(ROOT))
    for path in sorted(FIGURES.glob("*.png")):
        print(" ", path.relative_to(ROOT))

    agreement_path = TABLES / "annotator_agreement.csv"
    if not agreement_path.exists():
        print(
            "\nNOTE: annotator_agreement.csv was not generated. "
            "Both annotators must complete the 150-row sheet first, "
            "then run: python3 scripts/08_human_labels.py --evaluate"
        )


if __name__ == "__main__":
    main()
