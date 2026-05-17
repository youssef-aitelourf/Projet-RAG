import re
from dataclasses import dataclass
from typing import Literal

ChunkStrategy = Literal["fixed", "recursive", "sentence"]


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: str


def chunk_text(
    text: str,
    source: str,
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    if strategy == "fixed":
        return _fixed(text, source, chunk_size, overlap)
    elif strategy == "recursive":
        return _recursive(text, source, chunk_size, overlap)
    else:
        return _sentence(text, source, chunk_size)


def _make_id(source: str, idx: int) -> str:
    return f"{source}__{idx}"


def _fixed(text: str, source: str, size: int, overlap: int) -> list[Chunk]:
    chunks, i = [], 0
    while i < len(text):
        piece = text[i : i + size].strip()
        if piece:
            chunks.append(Chunk(text=piece, source=source, chunk_id=_make_id(source, i)))
        i += size - overlap
    return chunks


def _recursive(text: str, source: str, size: int, overlap: int) -> list[Chunk]:
    separators = ["\n\n", "\n", ". ", " "]
    raw = _split_rec(text, size, separators)
    # merge small consecutive pieces with overlap
    merged, buf = [], ""
    for piece in raw:
        if len(buf) + len(piece) + 1 <= size:
            buf = (buf + " " + piece).strip() if buf else piece
        else:
            if buf:
                merged.append(buf)
            buf = piece
    if buf:
        merged.append(buf)
    return [Chunk(text=t, source=source, chunk_id=_make_id(source, i)) for i, t in enumerate(merged) if t.strip()]


def _split_rec(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size or not seps:
        return [text] if text.strip() else []
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    result = []
    for part in parts:
        if len(part) <= size:
            result.append(part)
        else:
            result.extend(_split_rec(part, size, rest))
    return result


def _sentence(text: str, source: str, max_size: int) -> list[Chunk]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, buf, idx = [], "", 0
    for sent in sentences:
        if len(buf) + len(sent) + 1 <= max_size:
            buf = (buf + " " + sent).strip() if buf else sent
        else:
            if buf:
                chunks.append(Chunk(text=buf, source=source, chunk_id=_make_id(source, idx)))
                idx += 1
            buf = sent
    if buf:
        chunks.append(Chunk(text=buf, source=source, chunk_id=_make_id(source, idx)))
    return chunks
