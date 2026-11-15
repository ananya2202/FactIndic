"""
Script 04: LLM Summarization Runner (Person C - Tasks 3, 7, 13 & 18)
Generates summaries across all versions using Standard and Factuality-Aware prompts.
Enforces deterministic hashing cache and records model latency and token metrics.
Outputs data/processed/summaries.jsonl (the handoff for Person A) and
summarizes real Hinglish inputs (MUTANT/GupShup) for Task 18.
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.loader import load_jsonl, save_jsonl
from factindic.llm.client import get_llm_client
from factindic.llm.prompts import FACTUALITY_AWARE_PROMPT, STANDARD_PROMPT, format_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate summaries using Standard and Factuality-Aware prompts.")
    parser.add_argument("--versions", default="data/processed/versions.jsonl", help="Input versions JSONL")
    parser.add_argument("--output", default="data/processed/summaries.jsonl", help="Output summaries JSONL")
    parser.add_argument("--real_inputs", default="data/raw/real_hinglish_sample.jsonl", help="Real Hinglish inputs JSONL")
    parser.add_argument("--real_output", default="data/processed/real_hinglish_summaries.jsonl", help="Real Hinglish summaries JSONL")
    parser.add_argument("--provider", default=None, help="LLM Provider: gemini, openai, mock (auto-detects from .env)")
    parser.add_argument("--model", default=None, help="Model name identifier (e.g. gemini-1.5-flash)")
    parser.add_argument("--cache", default="data/cache/llm_cache.json", help="Cache file path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on versions to process")
    args = parser.parse_args()

    versions_path = PROJECT_ROOT / args.versions
    output_path = PROJECT_ROOT / args.output
    cache_path = PROJECT_ROOT / args.cache
    real_inputs_path = PROJECT_ROOT / args.real_inputs
    real_output_path = PROJECT_ROOT / args.real_output

    print(f"[Person C - Step 5] Loading versions from {versions_path}...")
    versions = load_jsonl(str(versions_path))
    if args.limit:
        versions = versions[:args.limit]
    print(f"Loaded {len(versions)} version texts across mixing levels.")

    client = get_llm_client(provider=args.provider, model_name=args.model, cache_path=str(cache_path))

    summaries = []
    prompts_to_run = [
        ("standard", STANDARD_PROMPT),
        ("factuality_aware", FACTUALITY_AWARE_PROMPT),
    ]

    total_runs = len(versions) * len(prompts_to_run)
    print(f"Running LLM inference for {total_runs} combinations ({len(versions)} versions x 2 prompt types)...")

    for idx, v in enumerate(versions):
        art_id = v["article_id"]
        ver_id = v["version_id"]
        mixing_level = v.get("mixing_target", 0.0)
        input_text = v["text"]

        for prompt_name, prompt_template in prompts_to_run:
            formatted_prompt = format_prompt(prompt_template, input_text=input_text)
            gen_res = client.generate(prompt=formatted_prompt, text=input_text)
            summary_text = gen_res["text"].strip()

            if idx < 3 or (idx + 1) % 100 == 0 or len(versions) <= 5:
                print(f"  [{idx+1}/{len(versions)}] ({v.get('language_label', 'mixed')}) {prompt_name} -> \"{summary_text[:60]}...\" ({gen_res.get('latency_ms', 0)}ms)")

            summaries.append({
                "article_id": art_id,
                "version_id": ver_id,
                "mixing_level": mixing_level,
                "language_label": v.get("language_label", f"{int(mixing_level*100)}% mixed"),
                "prompt_type": prompt_name,
                "round": 0,
                "summary": summary_text,
                "word_count": len(summary_text.split()),
                "model": gen_res.get("model", args.model),
                "cached": gen_res.get("cached", False),
                "latency_ms": gen_res.get("latency_ms", 0),
            })

    save_jsonl(summaries, str(output_path))
    print(f"\n[Person C - Step 5 Complete] Successfully generated {len(summaries)} summaries.")
    print(f"Handoff saved to: {output_path}")

    # Task 18: Summarize real Hinglish inputs (MUTANT & GupShup)
    if real_inputs_path.exists():
        print(f"\n[Person C - Task 18] Summarizing real Hinglish inputs from {real_inputs_path}...")
        real_inputs = load_jsonl(str(real_inputs_path))
        real_summaries = []
        for item in real_inputs:
            r_id = item.get("id")
            source_type = item.get("source", "Real_Hinglish")
            r_text = item.get("text", "")

            for prompt_name, prompt_template in prompts_to_run:
                formatted_prompt = format_prompt(prompt_template, input_text=r_text)
                gen_res = client.generate(prompt=formatted_prompt, text=r_text)
                real_summaries.append({
                    "article_id": r_id,
                    "version_id": f"{r_id}_real",
                    "source_corpus": source_type,
                    "prompt_type": prompt_name,
                    "round": 0,
                    "text": r_text,
                    "summary": gen_res["text"].strip(),
                    "reference_summary": item.get("reference_summary", ""),
                    "word_count": len(gen_res["text"].strip().split()),
                    "latency_ms": gen_res.get("latency_ms", 0),
                })
        save_jsonl(real_summaries, str(real_output_path))
        print(f"Saved {len(real_summaries)} real Hinglish summaries to: {real_output_path}")

    print(f"Response cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
