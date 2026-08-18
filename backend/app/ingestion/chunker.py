"""
Simple recursive character-based text splitter. No extra dependency (e.g.
langchain) -- just enough logic to split on paragraph/sentence boundaries
where possible, with a character overlap between consecutive chunks so
context isn't lost at chunk boundaries.
"""
from app.core.config import settings

SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    pieces = _split(text, SEPARATORS)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current:
            chunks.append(current.strip())
        if len(piece) > chunk_size:
            # piece itself too big (no smaller separator worked) -> hard split
            for i in range(0, len(piece), chunk_size - overlap):
                chunks.append(piece[i : i + chunk_size].strip())
            current = ""
        else:
            current = piece

    if current.strip():
        chunks.append(current.strip())

    # apply overlap by prepending the tail of the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return [c for c in chunks if c]



def _split(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split(text, rest)
    parts = text.split(sep)
    return [p + sep for p in parts[:-1]] + [parts[-1]]
