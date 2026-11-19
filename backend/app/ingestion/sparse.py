"""
Shared sparse-vector (BM25) utilities.

WHY THIS FILE EXISTS
====================
Qdrant supports hybrid search (dense vector + sparse/BM25 + Reciprocal Rank
Fusion) natively. To use it we must store a *sparse vector* next to each chunk's
dense embedding. A sparse vector is a dict of {token_id: weight} where only the
tokens that actually appear in the chunk are non-zero -- unlike a dense vector
which has a value for every dimension.

We compute these BM25-style term-frequency weights ourselves (pure Python, CPU,
offline). Doing the token->id hashing here -- and using THIS SAME module at BOTH
ingestion time (build the index) and query time (build the query vector) --
guarantees the token->id mapping is consistent, which is essential for the
keyword match to work.

This is intentionally dependency-free (stdlib only): tokenization is a simple
regex, and term hashing uses Python's built-in hash of a normalized token
modulo the sparse vector dimension. No NLTK/spaCy needed.
This keeps ingestion fast and avoids extra installs.
"""
import re
from collections import Counter
from typing import Iterable

# Sparse vector dimensionality. Qdrant uses 64-bit; we hash tokens into this
# space. 30000 is plenty to keep collisions low while staying small enough for
# efficient scans. Overlap between a chunk token and a query token only matters
# if they hash to the same id -- a rare collision here just slightly boosts a
# match, never breaks it.
SPARSE_DIM = 30000

# Tokenizer: keep word-ish tokens (letters, digits, common punctuation runs like
# account codes "001234", "CAD", "VOSTRO_ACCOUNT", "CR/DR", "10.5"). Lowercase
# them so "SAVINGS" and "savings" match. We keep underscores and dots/hyphens
# inside tokens because Finacle field names / codes often contain them
# (e.g. "ACCOUNT_OPEN_DATE", "chq.available").
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:[.\-/][A-Za-z0-9_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Split text into normalized tokens (lowercased word spans)."""
    return _TOKEN_RE.findall(text.lower())


def _token_to_id(token: str) -> int:
    """Hash a normalized token into the sparse-vector id space.

    Uses a stable, non-negative hash (MurmurHash2-style via Python's builtin
    hash on bytes, then normalized). Python's built-in hash() is randomized per
    process for str by default, so we hash the UTF-8 *bytes* with a fixed seed to
    get a deterministic value across processes/restarts. This matters because the
    ingestion backend and the query backend may run in different processes.
    """
    # FNV-1a 32-bit over the token bytes -- simple, deterministic, well-distributed.
    h = 2166136261
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h % SPARSE_DIM


def bm25_term_frequencies(tokens: list[str], k1: float = 1.2, b: float = 0.75) -> dict[int, float]:
    """Convert a bag of tokens into {token_id: BM25-ish tf weight}.

    We use the standard BM25 term-frequency component (without document-frequency
    normalization -- Qdrant's internal sparse scoring handles IDF corpus-wide):
        w(t) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avg_dl)))

    For a per-chunk query we don't have corpus statistics at query time, so we
    plug in a simple self-normalizing length factor. This gives higher weight to
    tokens repeated within the chunk while down-weighting every-token-in-a-long-
    chunk dilution. Good enough for precision-focused keyword matching.

    Arguments
    ---------
    tokens: normalized tokens from `tokenize()`.
    k1, b:  standard BM25 saturation / length-normalization constants.
    """
    counts = Counter(tokens)
    length = max(len(tokens), 1)  # avoid div-by-zero on empty
    dl_ratio = length / 200.0     # heuristic average document length (200 tokens)
    out: dict[int, float] = {}
    for token, tf in counts.items():
        tf = float(tf)
        denom = tf + k1 * (1.0 - b + b * dl_ratio)
        weight = (tf * (k1 + 1.0)) / denom if denom else 0.0
        tid = _token_to_id(token)
        out[tid] = out.get(tid, 0.0) + weight  # merge rare hash collisions
    return out


def build_sparse_query(text: str) -> dict[int, float]:
    """Build a sparse query vector from a raw question/term string.

    Queries are short, so we reuse the same term-frequency builder. Stop the
    query from being empty-weighted.
    """
    return bm25_term_frequencies(tokenize(text))
