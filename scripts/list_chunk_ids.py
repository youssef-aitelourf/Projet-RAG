#!/usr/bin/env python3
"""List chunk IDs produced by the default recursive chunking strategy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_dir, load_file


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/list_chunk_ids.py <file_or_directory> [strategy]")
        sys.exit(1)
    path = Path(sys.argv[1])
    strategy = sys.argv[2] if len(sys.argv) > 2 else "recursive"
    chunks = load_file(path, strategy=strategy) if path.is_file() else load_dir(path, strategy=strategy)
    for c in chunks:
        preview = c.text.replace("\n", " ")[:100]
        print(f"{c.chunk_id}\t[{c.source}]\t{preview}")


if __name__ == "__main__":
    main()
