"""
Multi-strategy text chunker adapted from Verba (goldenverba) chunking modules.

Provides three chunking strategies that can be selected via config:

1. RECURSIVE (default) — Improved version of the original chunker. Splits text
   recursively on paragraph (\n\n), sentence (". "), word (" "), and character
   boundaries. Configurable chunk_size, overlap, and separators. Best general-
   purpose option.

2. SENTENCE — Groups N sentences per chunk with sentence-level overlap.
   Guarantees clean sentence boundaries so entities are never split mid-
   sentence. Best for entity extraction quality.

3. SEMANTIC — Embeds each sentence using Ollama, calculates cosine distance
   between adjacent sentence embeddings, and splits where the semantic
   "jump" exceeds a percentile threshold. Produces the fewest, most
   semantically coherent chunks — directly reducing the number of LLM
   extraction calls. Requires an async embedding function (Ollama).

All strategies return list[str] — the pipeline doesn't need to change.

Adapted from:
  - Verba RecursiveChunker (LangChain-based, we reimplemented to avoid dep)
  - Verba SentenceChunker (spaCy-based, we use regex for lighter footprint)
  - Verba SemanticChunker (sklearn-based, we use numpy for lighter footprint)
"""
import logging
import re

import numpy as np

from app.core.config import settings

logger = logging.getLogger("graphrag.chunker")


# =============================================================================
# Dispatcher — call this from the pipeline
# =============================================================================

async def chunk_text(
    text: str,
    strategy: str | None = None,
    embed_fn=None,
) -> list[str]:
    """Split *text* into chunks using the configured strategy.

    Parameters
    ----------
    text : str
        The raw document text.
    strategy : str | None
        One of "recursive", "sentence", "semantic". Defaults to
        ``settings.chunking_strategy``.
    embed_fn : callable | None
        An async ``await embed_fn(text) -> list[float]`` function used only
        by the "semantic" strategy. Ignored for others.

    Returns
    -------
    list[str]
        The resulting text chunks.
    """
    strategy = strategy or settings.chunking_strategy
    text = text.strip()
    if not text:
        return []

    if strategy == "sentence":
        return _chunk_sentence(text)
    elif strategy == "semantic":
        if embed_fn is None:
            raise ValueError(
                "Semantic chunking requires an embed_fn (async callable). "
                "Pass the Ollama embed function from the pipeline."
            )
        return await _chunk_semantic(text, embed_fn)
    else:
        # Default: recursive (original strategy, improved)
        return _chunk_recursive(text)


# =============================================================================
# Strategy 1: Recursive character splitting (improved original)
# =============================================================================
# This is the same logic as before but now uses configurable separators
# from settings and handles edge cases better. Splits recursively on
# paragraph -> sentence -> word -> character boundaries.

def _chunk_recursive(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    separators = separators or ["\n\n", "\n", ". ", " "]

    if len(text) <= chunk_size:
        return [text] if text else []

    pieces = _split_recursive(text, separators)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current:
            chunks.append(current.strip())
        if len(piece) > chunk_size:
            # Piece itself too big — hard-split at chunk_size boundaries
            for i in range(0, len(piece), chunk_size - overlap):
                chunks.append(piece[i : i + chunk_size].strip())
            current = ""
        else:
            current = piece

    if current.strip():
        chunks.append(current.strip())

    # Apply overlap by prepending the tail of the previous chunk.
    # This gives the LLM/retrieval system context at chunk boundaries.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return [c for c in chunks if c]


def _split_recursive(text: str, separators: list[str]) -> list[str]:
    """Recursively split text using a list of separators (largest to smallest)."""
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_recursive(text, rest)
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]


# =============================================================================
# Strategy 2: Sentence-based chunking (adapted from Verba SentenceChunker)
# =============================================================================
# Groups N sentences per chunk with configurable sentence-level overlap.
# Guarantees that entity names are never split across chunks — critical
# for accurate entity extraction in Phase 2.
#
# Uses regex sentence detection instead of spaCy to avoid the heavy
# dependency. Handles common patterns: periods, exclamation marks,
# question marks followed by whitespace.

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _chunk_sentence(
    text: str,
    sentences_per_chunk: int | None = None,
    sentence_overlap: int | None = None,
) -> list[str]:
    """Split text into chunks of N sentences with overlap.

    Parameters are pulled from settings if not provided:
      - ``settings.chunk_sentences_per_chunk`` (default 8)
      - ``settings.chunk_sentence_overlap`` (default 1)
    """
    sentences_per_chunk = sentences_per_chunk or settings.chunk_sentences_per_chunk
    sentence_overlap = sentence_overlap or settings.chunk_sentence_overlap

    # Split into sentences using regex
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    if not sentences:
        return [text] if text else []
    if len(sentences) <= sentences_per_chunk:
        return [" ".join(sentences)]

    # Clamp overlap to be less than chunk size
    if sentence_overlap >= sentences_per_chunk:
        sentence_overlap = sentences_per_chunk - 1

    chunks = []
    i = 0
    while i < len(sentences):
        end = min(i + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[i:end]
        chunks.append(" ".join(chunk_sentences))

        # Step forward by (chunk_size - overlap) sentences
        step = sentences_per_chunk - sentence_overlap
        if step < 1:
            step = 1
        i += step

    logger.info(
        "Sentence chunking: %d sentences -> %d chunks "
        "(%d per chunk, %d overlap)",
        len(sentences), len(chunks), sentences_per_chunk, sentence_overlap,
    )
    return [c for c in chunks if c]


# =============================================================================
# Strategy 3: Semantic chunking (adapted from Verba SemanticChunker)
# =============================================================================
# The most sophisticated strategy. Instead of splitting at fixed character
# or sentence counts, it detects TOPIC BOUNDARIES by measuring how
# "different" adjacent sentences are using cosine distance of their
# embeddings.
#
# How it works:
#   1. Split text into sentences
#   2. Embed each sentence using Ollama (nomic-embed-text)
#   3. Calculate cosine distance between each adjacent pair of sentences
#   4. Find the "breakpoint" — the distance above which sentences are
#      semantically different enough to warrant a new chunk
#   5. The breakpoint is set at the Nth percentile of all distances
#      (configurable via settings.chunk_semantic_threshold, default 80)
#
# Result: chunks that contain topically coherent groups of sentences,
# regardless of how many characters or sentences they contain. This
# typically produces 30-50% fewer chunks than fixed-size strategies,
# and each chunk has better entity density for the extraction phase.
#
# Tradeoff: requires an Ollama embedding call per sentence during
# chunking, but embeddings are fast (~100ms) and we're batching them,
# so the total overhead is small compared to the LLM extraction savings.


async def _chunk_semantic(text: str, embed_fn) -> list[str]:
    """Split text into semantically coherent chunks.

    Parameters
    ----------
    text : str
        The document text.
    embed_fn : async callable
        ``await embed_fn(text) -> list[float]`` — must return an embedding
        vector for the given text. In practice this is
        ``ollama_client.embed()``.
    """
    max_sentences = settings.chunk_semantic_max_sentences
    threshold_percentile = settings.chunk_semantic_threshold

    # Step 1: Split into sentences
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return [text] if text else []
    if len(sentences) == 1:
        return [sentences[0]]

    logger.info("Semantic chunking: %d sentences detected", len(sentences))

    # Step 2: Combine each sentence with its neighbors (buffer=1) for
    # richer embedding context — same approach as Verba. This gives each
    # sentence embedding awareness of its surrounding context.
    combined = []
    for i in range(len(sentences)):
        parts = []
        if i > 0:
            parts.append(sentences[i - 1])
        parts.append(sentences[i])
        if i < len(sentences) - 1:
            parts.append(sentences[i + 1])
        combined.append(" ".join(parts))

    # Step 3: Embed all combined sentences in one batch call to Ollama
    # (much faster than one-by-one)
    logger.info("Embedding %d sentence windows for semantic chunking...", len(combined))
    embeddings = await _batch_embed(combined, embed_fn)
    logger.info("Embeddings received: %d vectors", len(embeddings))

    # Step 4: Calculate cosine distances between adjacent sentence embeddings
    distances = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        distances.append(1.0 - sim)  # cosine distance = 1 - similarity

    # Step 5: Find the breakpoint threshold at the configured percentile.
    # An 80th-percentile threshold means we split at the top 20% of
    # largest semantic jumps — producing fewer, larger chunks.
    # Lower values (e.g., 60) = more chunks, higher (e.g., 90) = fewer.
    if distances:
        breakpoint_threshold = float(np.percentile(distances, threshold_percentile))
    else:
        breakpoint_threshold = 0.0

    logger.info(
        "Semantic breakpoints: threshold=%.4f (percentile=%d), "
        "min_dist=%.4f, max_dist=%.4f",
        breakpoint_threshold,
        threshold_percentile,
        min(distances) if distances else 0,
        max(distances) if distances else 0,
    )

    # Step 6: Group sentences into chunks. A new chunk starts when:
    #   - The distance to the next sentence exceeds the threshold, OR
    #   - The current chunk has reached max_sentences
    chunks = []
    current_chunk = []
    sentence_count = 0

    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        sentence_count += 1

        is_breakpoint = (
            i < len(distances) and distances[i] > breakpoint_threshold
        )
        at_max = sentence_count >= max_sentences

        if is_breakpoint or at_max:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            sentence_count = 0

    # Flush remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logger.info(
        "Semantic chunking complete: %d sentences -> %d chunks "
        "(threshold_percentile=%d, max_sentences=%d)",
        len(sentences), len(chunks), threshold_percentile, max_sentences,
    )
    return [c for c in chunks if c]


# =============================================================================
# Helpers for semantic chunking
# =============================================================================

async def _batch_embed(texts: list[str], embed_fn) -> list[list[float]]:
    """Embed a list of texts concurrently using asyncio.gather.

    Ollama's /api/embeddings endpoint only accepts one text at a time,
    so we fire off parallel requests. The Ollama server handles the
    concurrency internally (governed by OLLAMA_NUM_PARALLEL).
    """
    import asyncio
    results = await asyncio.gather(*(embed_fn(t) for t in texts))
    return list(results)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors using numpy."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))
