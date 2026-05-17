#!/usr/bin/env python3
"""Verify that core dependencies import correctly."""

import sys

MODULES = [
    "chromadb",
    "sentence_transformers",
    "rouge_score",
    "rank_bm25",
    "click",
    "openai",
    "pandas",
    "numpy",
]


def main() -> int:
    failed = []
    for name in MODULES:
        try:
            __import__(name)
            print(f"  ok  {name}")
        except ImportError as e:
            print(f"  FAIL {name}: {e}")
            failed.append(name)
    if failed:
        print(f"\nMissing modules: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return 1
    print("\nAll imports OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
