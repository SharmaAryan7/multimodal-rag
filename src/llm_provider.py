"""
LLM provider — caches Gemini LLM instances so we don't recreate them on every query.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import (
    LLM_MODEL,
    REWRITER_MODEL,
    LLM_TEMPERATURE,
)

_llm_instance = None
_rewriter_instance = None


def get_llm():
    """Get the cached main Gemini LLM instance."""
    global _llm_instance

    if _llm_instance is None:
        print(f"   [LLM] Initializing {LLM_MODEL} (cached)...")

        _llm_instance = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=LLM_TEMPERATURE,
        )

    return _llm_instance


def get_rewriter_llm():
    """Get the cached Gemini rewriter."""
    global _rewriter_instance

    if _rewriter_instance is None:
        print(f"   [LLM] Initializing rewriter {REWRITER_MODEL} (cached)...")

        _rewriter_instance = ChatGoogleGenerativeAI(
            model=REWRITER_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

    return _rewriter_instance


def warmup_models():
    """Pre-load Gemini models."""
    print("   [LLM] Warming up Gemini...")
    get_llm()
    get_rewriter_llm()
    print("   [LLM] Models ready.")