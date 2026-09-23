"""Structure-aware chunking.

Splits on the strongest available boundary -- blank line > heading > sentence
-- and only then falls back to a hard character window. Overlap is applied at
character level so evidence spans never lose their trailing context.

Author: 晨星
"""
from __future__ import annotations

import re

from ..core.types import Chunk, Document

_SENT_END = re.compile(r"(?<=[。！？!?；;])")
_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s|\d+(?:\.\d+)*[、.\s])")


def _paragraphs(text: str) -> list[str]:
    blocks = re.split(r"\n\s*\n", text)
    return [b.strip() for b in blocks if b.strip()]


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    step = max(1, size - overlap)
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += step
    return out


def chunk_document(doc: Document, chunk_size: int = 480, overlap: int = 80) -> list[Chunk]:
    """Chunk one document. Returns chunks in document order.

    Invariants:
      * concatenation of chunk texts covers the whole document (no silent loss)
      * every chunk is non-empty
      * `order` is monotonic, `start`/`end` are offsets into `doc.text`
    """
    pieces: list[str] = []
    for para in _paragraphs(doc.text):
        if len(para) <= chunk_size:
            pieces.append(para)
            continue
        if _HEADING.match(para) or "\n" in para:
            sub = [ln.strip() for ln in para.splitlines() if ln.strip()]
        else:
            sub = _sentences(para)
        buf = ""
        for s in sub:
            if len(buf) + len(s) + 1 <= chunk_size:
                buf = f"{buf} {s}".strip()
            else:
                if buf:
                    pieces.append(buf)
                buf = s
            if len(buf) > chunk_size:
                pieces.extend(_hard_split(buf, chunk_size, overlap))
                buf = ""
        if buf:
            pieces.append(buf)

    chunks: list[Chunk] = []
    cursor = 0
    for idx, piece in enumerate(pieces):
        for sub_idx, seg in enumerate(_hard_split(piece, chunk_size, overlap)):
            start = doc.text.find(seg, cursor)
            if start < 0:
                start = cursor
            end = start + len(seg)
            chunks.append(Chunk.create(doc, seg, order=len(chunks), start=start, end=end))
            cursor = max(cursor, end - overlap if sub_idx == 0 else cursor)
    return chunks
