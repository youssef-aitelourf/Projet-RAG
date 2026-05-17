"""Retrieval metrics that do not require an LLM judge."""


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant IDs found in the top-k retrieved IDs."""
    if not relevant_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for rid in relevant_ids if rid in top)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean reciprocal rank of the first relevant chunk (single query)."""
    if not relevant_ids:
        return 0.0
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def extract_retrieved_ids(docs: list[dict]) -> list[str]:
    """Get chunk IDs from retrieved docs, falling back to text hash key."""
    ids = []
    for d in docs:
        if cid := d.get("chunk_id"):
            ids.append(cid)
        elif text := d.get("text"):
            ids.append(text[:200])
    return ids
