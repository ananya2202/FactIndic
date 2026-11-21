"""
FactIndic Interactive Studio & Benchmark Explorer
Clean, modern, and minimalist web interface for evaluating and revising code-mixed Indic summaries.
Run locally via: streamlit run app.py
"""

import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from factindic.data.loader import load_jsonl
from factindic.llm.client import get_llm_client
from factindic.llm.prompts import FACTUALITY_AWARE_PROMPT, STANDARD_PROMPT, format_prompt
from factindic.llm.reviser import SummaryReviser
from factindic.verify.verifier import FactualityVerifier

# Page Configuration
st.set_page_config(
    page_title="FactIndic",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern Minimalist Styling
st.markdown("""
<style>
    /* Clean line typography and cards */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    .clean-card {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
        background-color: rgba(128, 128, 128, 0.03);
    }
    .badge-tag {
        display: inline-block;
        font-family: monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    .badge-number { background-color: rgba(234, 179, 8, 0.15); color: #ca8a04; }
    .badge-date { background-color: rgba(59, 130, 246, 0.15); color: #2563eb; }
    .badge-entity { background-color: rgba(168, 85, 247, 0.15); color: #9333ea; }
    .badge-relation { background-color: rgba(14, 165, 233, 0.15); color: #0284c7; }
    .badge-supported { background-color: rgba(34, 197, 94, 0.15); color: #16a34a; }
    .badge-unsupported { background-color: rgba(239, 68, 68, 0.15); color: #dc2626; }
    .stat-label { font-size: 0.75rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .stat-val { font-size: 1.35rem; font-weight: 700; font-family: monospace; margin-top: 0.2rem; }
    pre code, pre {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache Verifier to ensure fast UI responses
@st.cache_resource
def get_verifier(use_hf: bool = False):
    return FactualityVerifier(use_hf=use_hf)

def get_active_client(provider_name: str, api_key: str = ""):
    if api_key:
        import os
        os.environ["GEMINI_API_KEY"] = api_key
    return get_llm_client(provider=provider_name)

@st.cache_data
def load_sample_versions():
    vpath = PROJECT_ROOT / "data" / "processed" / "versions.jsonl"
    if vpath.exists():
        return load_jsonl(str(vpath))
    return []

@st.cache_data
def load_metrics_df():
    mpath = PROJECT_ROOT / "results" / "tables" / "metrics_summary.csv"
    if mpath.exists():
        return pd.read_csv(mpath)
    return None

versions = load_sample_versions()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### FactIndic")
    st.caption("Factual Consistency in Code-Mixed Summaries")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Live Verification Studio", "Benchmark & Analysis Plots", "Dataset & Verdicts Browser"],
        index=0,
    )
    st.markdown("---")
    st.markdown("##### LLM Backend")
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_API_KEY") != "YOUR_GEMINI_API_KEY_HERE")
    provider_options = ["Gemini API (Live Neural LLM)", "Offline Simulator (Mock)"]
    chosen_provider_ui = st.selectbox(
        "Generation Model:",
        provider_options,
        index=0 if has_gemini else 1,
        help="Use real Gemini API for organic, un-scripted generation or the offline simulator."
    )
    provider_key = "gemini" if "Gemini" in chosen_provider_ui else "mock"
    user_api_key = ""
    if provider_key == "gemini" and not has_gemini:
        user_api_key = st.text_input("Gemini API Key:", type="password", help="Enter key to enable live Gemini generation")
    
    st.caption(f"Active Provider: `{provider_key.upper()}`")

client = get_active_client(provider_key, user_api_key)
verifier = get_verifier(use_hf=False)

# PAGE 1: LIVE INTERACTIVE VERIFICATION STUDIO
if page == "Live Verification Studio":
    st.title("Live Verification & Revision Studio")
    st.caption("Atomic claims decomposition, rule and NLI verification, and iterative revision on Hinglish text.")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("##### 1. Source Context (Hindi / Hinglish)")
        sample_options = ["Custom Input"] + [f"{v['article_id']} ({v.get('language_label', 'Hinglish')})" for v in versions[:10]]
        chosen_sample = st.selectbox("Select Sample or Type Custom:", sample_options, index=1 if len(sample_options) > 1 else 0)

        if chosen_sample != "Custom Input":
            idx = sample_options.index(chosen_sample) - 1
            default_text = versions[idx]["text"]
            default_mix = versions[idx].get("mixing_target", 0.0)
        else:
            default_text = "शुक्रवार शाम को शुरु occurred फ़ेस्टिवल में इंसाफ़ का तराज़ू फ़िल्म दिखाई गई. उत्सव के during छोटी सी बात, नया दौर और वक़्त जैसी फ़िल्में भी दिखाई जाएंगी."
            default_mix = 0.3

        source_text = st.text_area("Source Article Context:", value=default_text, height=150)

    with col_r:
        st.markdown("##### 2. Generation Configuration")
        prompt_choice = st.radio(
            "Prompt Strategy:",
            ["Standard Prompt", "Factuality-Aware Prompt"],
            help="Factuality-aware prompting instructs the model to omit ungrounded assertions."
        )
        is_factual = (prompt_choice == "Factuality-Aware Prompt")

        # Reset active summary if sample or prompt strategy changed
        if st.session_state.get("prev_sample") != chosen_sample or st.session_state.get("prev_prompt") != prompt_choice:
            st.session_state["prev_sample"] = chosen_sample
            st.session_state["prev_prompt"] = prompt_choice
            st.session_state.pop("active_summary", None)
            st.session_state.pop("last_prompt_sent", None)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Summary", type="primary", use_container_width=True):
            p_template = FACTUALITY_AWARE_PROMPT if is_factual else STANDARD_PROMPT
            formatted_p = format_prompt(p_template, input_text=source_text)
            try:
                gen_res = client.generate(prompt=formatted_p, text=source_text, force_refresh=True)
                st.session_state["active_summary"] = gen_res["text"].strip()
                st.session_state["last_prompt_sent"] = formatted_p
            except Exception as e:
                st.error(f"Generation error: {e}")

    # Prompt Comparison Viewer
    with st.expander("Compare Prompts: Standard vs Factuality-Aware", expanded=False):
        pt1, pt2 = st.columns(2)
        with pt1:
            st.markdown("**Standard Prompt Template:**")
            st.code(STANDARD_PROMPT, language="markdown")
        with pt2:
            st.markdown("**Factuality-Aware Prompt Template:**")
            st.code(FACTUALITY_AWARE_PROMPT, language="markdown")
        
        if "last_prompt_sent" in st.session_state:
            st.markdown("---")
            st.markdown("**Exact Formatted Prompt Sent to LLM:**")
            st.code(st.session_state["last_prompt_sent"], language="markdown")

    st.markdown("##### 3. Summary Text")
    summary_val = st.session_state.get("active_summary", "")
    summary_input = st.text_area(
        "Summary Text to Verify (Click 'Generate Summary' or edit directly):",
        value=summary_val,
        height=95,
        placeholder="No summary generated yet. Click 'Generate Summary' above.",
    )

    if not summary_input.strip():
        st.info("Select a source article and click 'Generate Summary' (or type a summary) to run verification.")
        st.stop()

    # Run Verification only if summary is present
    res = verifier.verify_summary(source_text, summary_input)

    st.markdown("---")
    st.markdown("##### 4. Verification Output & Factuality Metrics")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown('<div class="clean-card"><div class="stat-label">Atomic Claims</div><div class="stat-val">' + str(res["total_claims"]) + '</div></div>', unsafe_allow_html=True)
    with m2:
        hr_color = "#dc2626" if res["hallucination_rate"] > 0 else "#16a34a"
        st.markdown(f'<div class="clean-card"><div class="stat-label">Hallucination Rate</div><div class="stat-val" style="color: {hr_color}">{res["hallucination_rate"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="clean-card"><div class="stat-label">Factual Recall</div><div class="stat-val" style="color: #16a34a">{res["factual_recall"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="clean-card"><div class="stat-label">Weighted Score</div><div class="stat-val">{res["weighted_score"]:.2f}</div></div>', unsafe_allow_html=True)

    # Display Atomic Claims Breakdown
    st.markdown("###### Atomic Claims & Typed Verdicts:")
    for c in res["claims"]:
        badge_cls = f"badge-{c['type']}"
        is_supp = (c["verdict"] == "supported")
        status_cls = "badge-supported" if is_supp else "badge-unsupported"
        status_txt = "Supported" if is_supp else c['error_type'].replace('_', ' ').title()

        with st.container():
            st.markdown(f"""
            <div class="clean-card" style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge-tag {badge_cls}">{c['type']}</span>
                    <span style="font-family: monospace; font-size: 0.9rem; margin-left: 0.5rem;">{c['text']}</span>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">
                        NLI Label: <b>{c['nli_label']}</b> | Confidence Score: <b>{c['nli_score']:.2f}</b> | Rules Passed: <b>{c['rule_passed']}</b>
                    </div>
                </div>
                <span class="badge-tag {status_cls}">{status_txt}</span>
            </div>
            """, unsafe_allow_html=True)

    # Iterative Revision Trigger
    if res["unsupported_claims"] > 0:
        st.markdown("---")
        st.warning(f"Found {res['unsupported_claims']} unsupported claim(s). Trigger the revision loop to remove/correct ungrounded facts.")
        if st.button("Run Iterative Revision Loop", type="secondary"):
            try:
                reviser = SummaryReviser(llm_client=client, max_rounds=2)
                flagged = [c["text"] for c in res["claims"] if c["verdict"] == "unsupported"]
                rev_out = reviser.revise_summary(source_text, summary_input, flagged, current_round=1)
                
                re_ver = verifier.verify_summary(source_text, rev_out["revised_summary"])
                st.success("Revision Loop Complete")
                st.markdown(f"**Revised Summary (Round 1):** `{rev_out['revised_summary']}`")
                st.markdown(f"**Hallucination Rate after Revision:** **{re_ver['hallucination_rate']*100:.1f}%** (Resolved: {re_ver['unsupported_claims'] == 0})")
            except Exception as e:
                st.error(f"Revision error: {e}")

# PAGE 2: BENCHMARK & ANALYSIS PLOTS
elif page == "Benchmark & Analysis Plots":
    st.title("Benchmark & Analysis Plots")
    st.caption("Visualizing empirical findings across 2,500 summaries and 5 code-mixing levels (0%, 10%, 30%, 50%, 100%).")

    fig_dir = PROJECT_ROOT / "results" / "figures"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 1. Prompt Strategy Comparison")
        st.caption("Standard prompt vs Factuality-Aware prompt across mixing tiers.")
        p1 = fig_dir / "prompt_comparison.png"
        if p1.exists():
            st.image(str(p1), use_container_width=True)
            st.info("Factuality-Aware prompting eliminates hallucinations (~0.00) while standard prompt peaks on Hinglish (10.4%).")

    with c2:
        st.markdown("##### 2. Iterative Revision Progression")
        st.caption("Hallucination reduction across Round 0 → Round 1 → Round 2.")
        p2 = fig_dir / "revise_loop_progression.png"
        if p2.exists():
            st.image(str(p2), use_container_width=True)
            st.info("The iterative loop eliminates ungrounded claims after Round 1 without degrading content recall.")

    st.markdown("---")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### 3. Code-Mixing Vulnerability Curve")
        st.caption("Mean hallucination rate across 0% (Hindi) to 100% (English).")
        p3 = fig_dir / "hallucination_by_mixing_tier.png"
        if p3.exists():
            st.image(str(p3), use_container_width=True)
            st.info("Hallucination rate shows an elevated plateau across the code-mixed bands (10%-50%).")

    with c4:
        st.markdown("##### 4. Factuality Error Taxonomy")
        st.caption("Distribution of Number Drift, Date Alteration, Entity Swap, and Relation Distortion.")
        p4 = fig_dir / "error_distribution.png"
        if p4.exists():
            st.image(str(p4), use_container_width=True)
            st.info("Number Drift is the predominant failure mode in code-mixed generation, followed by Relation Distortion.")

# PAGE 3: DATASET & VERDICTS BROWSER
elif page == "Dataset & Verdicts Browser":
    st.title("Dataset & Verdicts Browser")
    st.caption("Explore verified summaries and error labels from the processed handoff records.")

    vpath = PROJECT_ROOT / "data" / "processed" / "verdicts.jsonl"
    if vpath.exists():
        records = load_jsonl(str(vpath))
        df = pd.DataFrame([
            {
                "article_id": r.get("article_id"),
                "mixing_level": f"{int(r.get('mixing_level', 0)*100)}%",
                "prompt_type": r.get("prompt_type"),
                "total_claims": r.get("total_claims"),
                "unsupported": r.get("unsupported_claims"),
                "hallucination_rate": f"{r.get('hallucination_rate', 0)*100:.1f}%",
                "factual_recall": f"{r.get('factual_recall', 0)*100:.1f}%",
                "summary": r.get("summary"),
            }
            for r in records
        ])

        tier_filter = st.multiselect("Filter by Mixing Tier:", ["0%", "10%", "30%", "50%", "100%"], default=["10%", "30%"])
        prompt_filter = st.multiselect("Filter by Prompt Type:", ["standard", "factuality_aware"], default=["standard", "factuality_aware"])

        filtered_df = df[df["mixing_level"].isin(tier_filter) & df["prompt_type"].isin(prompt_filter)]
        st.dataframe(filtered_df, use_container_width=True, height=450)
        st.caption(f"Showing {len(filtered_df)} of {len(df)} total evaluated summaries.")
    else:
        st.warning("Verdicts file (verdicts.jsonl) not found. Run scripts/05_verify.py to generate records.")
