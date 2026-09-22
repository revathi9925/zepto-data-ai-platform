import os
from typing import List, Dict

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def mock_classify_intent(query: str) -> str:
    """Keyword heuristic classification for Mock Mode."""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in KEYWORDS):
        return "policy_question"
    return "general_question"

def mock_retrieve_and_answer(query: str, chunks: List[Dict[str, str]]) -> dict:
    """Mock answer generator returning all retrieved source IDs."""
    top_chunk_snippet = chunks[0]["text"][:200]
    answer_text = f"Based on the retrieved context: {top_chunk_snippet}"
    return {
        "answer": answer_text,
        "sources": [c["id"] for c in chunks],
        "confidence": 1.0
    }

def mock_direct_answer() -> dict:
    """Mock direct answer generator for non-policy queries."""
    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
        "confidence": 1.0
    }