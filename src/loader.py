from pathlib import Path
from src.chunker import Chunk, chunk_text, ChunkStrategy

SUPPORTED = {".txt", ".md", ".pdf"}


def load_file(
    path: str | Path,
    strategy: ChunkStrategy = "recursive",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    path = Path(path)
    if path.suffix == ".pdf":
        text = _load_pdf(path)
    else:
        text = path.read_text(encoding="utf-8")
    return chunk_text(text, source=path.name, strategy=strategy, chunk_size=chunk_size, overlap=overlap)


def load_dir(dir_path: str | Path, **kwargs) -> list[Chunk]:
    chunks = []
    for f in sorted(Path(dir_path).iterdir()):
        if f.suffix in SUPPORTED:
            chunks.extend(load_file(f, **kwargs))
    return chunks


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(p.extract_text() or "" for p in reader.pages)
