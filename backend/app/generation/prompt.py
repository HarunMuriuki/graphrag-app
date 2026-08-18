"""Builds the final prompt sent to the LLM from retrieved chunks + graph facts."""
from app.retrieval.graph_search import facts_as_text

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using the provided "
    "context only. The context has two parts: TEXT PASSAGES retrieved by "
    "semantic search, and GRAPH FACTS derived from a knowledge graph built "
    "from the same documents, which may surface connections not stated "
    "verbatim in any single passage. If the context does not contain the "
    "answer, say so plainly instead of guessing."
)


def build_prompt(question: str, chunks: list[dict], graph_edges: list[dict]) -> str:
    passages = "\n\n".join(
        f"[Passage {i + 1} | source: {c.get('source', 'unknown')}]\n{c['text']}"
        for i, c in enumerate(chunks)
    ) or "(no passages retrieved)"

    facts = facts_as_text(graph_edges)
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "(no graph facts retrieved)"

    return f"""TEXT PASSAGES:
{passages}

GRAPH FACTS:
{facts_block}

QUESTION:
{question}

Answer the question using only the information above. Cite which passage(s) you used where relevant.
ANSWER:"""
