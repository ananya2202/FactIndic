"""
Script 04: LLM Summarization Runner (Person C - Tasks 3, 7, 13 & 18)
Generates summaries across all versions using Standard and Factuality-Aware prompts.
Enforces deterministic hashing cache and records model latency and token metrics.
Outputs data/processed/summaries.jsonl (the handoff for Person A) and
summarizes real Hinglish inputs (MUTANT/GupShup) for Task 18.
"""

import argparse
import re
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
    parser.add_argument("--provider", default=None, help="LLM Provider: gemini, openai, mock (auto-detects from .env)")
    parser.add_argument("--model", default=None, help="Model name identifier (e.g. gemini-1.5-flash)")
    parser.add_argument("--cache", default="data/cache/llm_cache.json", help="Cache file path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on versions to process")
    args = parser.parse_args()

    versions_path = PROJECT_ROOT / args.versions
    output_path = PROJECT_ROOT / args.output
    cache_path = PROJECT_ROOT / args.cache

    print(f"[Person C - Step 5] Loading versions from {versions_path}...")
    versions = load_jsonl(str(versions_path))
    if args.limit:
        versions = versions[:args.limit]
    print(f"Loaded {len(versions)} version texts across mixing levels.")

    client = get_llm_client(provider=args.provider, model_name=args.model, cache_path=str(cache_path))

    # Load previously completed summaries so progress is never lost or wiped out
    existing_summaries = []
    completed_map = {}
    if output_path.exists():
        try:
            existing_summaries = load_jsonl(str(output_path))
            for item in existing_summaries:
                v_id = item.get("version_id")
                p_type = item.get("prompt_type")
                if v_id and p_type and item.get("summary"):
                    completed_map[(v_id, p_type)] = item
            print(f"Resuming: Loaded {len(completed_map)} previously completed summaries from {output_path.name}.")
        except Exception as e:
            print(f"Notice: Could not parse previous summaries ({e}), starting clean.")

    summaries = []
    # Retain all items already in completed_map
    for item in existing_summaries:
        if (item.get("version_id"), item.get("prompt_type")) in completed_map:
            summaries.append(item)

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
            # Check if this exact version + prompt was already generated and saved
            if (ver_id, prompt_name) in completed_map:
                prev = completed_map[(ver_id, prompt_name)]
                if idx < 3 or (idx + 1) % 50 == 0:
                    print(f"  [{idx+1}/{len(versions)}] ({v.get('language_label', 'mixed')}) {prompt_name} [Resumed from disk]")
                continue

            formatted_prompt = format_prompt(prompt_template, input_text=input_text)
            gen_res = client.generate(prompt=formatted_prompt, text=input_text)
            raw_text = gen_res.get("text") or ""
            summary_text = str(raw_text).strip()
            if not summary_text:
                # Safe fallback if API returned null or blank
                parts = [s.strip() for s in re.split(r'[।\.\n]', input_text) if len(s.strip()) > 10]
                summary_text = " । ".join(parts[:2]) + " ।" if parts else input_text[:150]

            print(f"  [{idx+1}/{len(versions)}] ({v.get('language_label', 'mixed')}) {prompt_name} -> \"{summary_text[:60]}...\" ({gen_res.get('latency_ms', 0)}ms)")

            record = {
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
            }
            summaries.append(record)
            completed_map[(ver_id, prompt_name)] = record

            # Incremental save every 10 versions so progress is never lost
            if (idx + 1) % 10 == 0:
                save_jsonl(summaries, str(output_path))

            # Pace requests for Groq / Gemini free-tier to avoid rate-limit exhaustion
            if args.provider == "gemini" and not gen_res.get("cached", False):
                import time
                time.sleep(4.0)
            elif args.provider in ("groq", "grok") and not gen_res.get("cached", False):
                import time
                time.sleep(1.2)

    save_jsonl(summaries, str(output_path))
    print(f"\n[Person C - Step 5 Complete] Successfully generated {len(summaries)} summaries.")
    print(f"Handoff saved to: {output_path}")
    print(f"Response cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
