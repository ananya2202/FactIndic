"""Standard, Factuality-Aware, and Revision prompt templates for FactIndic."""

STANDARD_PROMPT = """You are an expert news summarizer. Write a concise, coherent summary of the provided text in 1 to 2 sentences (maximum 65 words). Maintain the dominant language and code-mixing style of the input.

INPUT TEXT:
{input_text}

SUMMARY:"""

FACTUALITY_AWARE_PROMPT = """You are a rigorous, factuality-constrained news summarizer. Your task is to produce a concise summary (maximum 65 words) that adheres strictly and faithfully to the input article.

CRITICAL FACTUALITY INSTRUCTIONS:
1. Every single claim, entity, number, percentage, and date MUST be explicitly stated in the input text.
2. Do NOT extrapolate, assume, speculate, or introduce external world knowledge.
3. If a number, person name, or location is mentioned, reproduce it with 100% exact fidelity.
4. Avoid ungrounded adjectives or embellishments.
5. If the input text is code-mixed (Hindi and English), preserve the code-mixing balance without translating the whole text.

INPUT TEXT:
{input_text}

FACTUAL SUMMARY:"""

REVISION_PROMPT = """You are revising a news summary. The previous summary contained factual inaccuracies or unsupported claims that were flagged during verification against the source article.

SOURCE ARTICLE:
{source_text}

PREVIOUS SUMMARY:
{previous_summary}

FLAGGED UNSUPPORTED CLAIMS:
{flagged_claims}

REVISION INSTRUCTIONS:
1. Rewrite the summary to completely REMOVE or CORRECT the flagged unsupported claims.
2. Retain all previously verified factual details.
3. Ensure the revised summary is strictly supported by the source article and remains under 65 words.
4. Maintain the dominant code-mixed language of the original text.

REVISED SUMMARY:"""


def format_prompt(template: str, **kwargs: str) -> str:
    """Formats a prompt template with provided variables."""
    return template.format(**kwargs)
