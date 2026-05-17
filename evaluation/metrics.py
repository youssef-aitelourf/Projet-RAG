from rouge_score import rouge_scorer

from evaluation.judge_cache import JudgeCache
from experiments.base import RAGResult
from src.llm import LLM

_cache: JudgeCache | None = None


def _get_cache() -> JudgeCache:
    global _cache
    if _cache is None:
        _cache = JudgeCache()
    return _cache

_FAITHFULNESS_PROMPT = """\
Rate how well the answer is GROUNDED in the context on a scale from 0.0 to 1.0.
- 1.0 = every claim is directly supported by the context
- 0.5 = partially supported, some unsupported claims
- 0.0 = contradicts or completely ignores the context
Output ONLY a single number. Nothing else.

Context:
{context}

Question: {question}
Answer: {answer}

Score:"""

_RELEVANCE_PROMPT = """\
Rate how relevant and complete the answer is to the question on a scale from 0.0 to 1.0.
- 1.0 = directly and completely answers the question
- 0.5 = partially answers
- 0.0 = off-topic or non-answer
Output ONLY a single number. Nothing else.

Question: {question}
Answer: {answer}

Score:"""

_CTX_PRECISION_PROMPT = """\
Does this passage contain information useful for answering the question?
Output ONLY "yes" or "no".

Question: {question}
Passage: {passage}

Answer:"""

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def faithfulness(llm: LLM, question: str, result: RAGResult) -> float:
    context = "\n\n".join(d["text"] for d in result.retrieved_docs)
    payload = f"{context[:3000]}|{result.answer}"
    cache = _get_cache()
    if (cached := cache.get("faithfulness", question, payload)) is not None:
        return cached
    raw = llm.ask(
        _FAITHFULNESS_PROMPT.format(
            context=context[:3000], question=question, answer=result.answer
        )
    )
    score = _parse_score(raw)
    cache.set("faithfulness", question, payload, score)
    return score


def answer_relevance(llm: LLM, question: str, answer: str) -> float:
    cache = _get_cache()
    if (cached := cache.get("relevance", question, answer)) is not None:
        return cached
    raw = llm.ask(_RELEVANCE_PROMPT.format(question=question, answer=answer))
    score = _parse_score(raw)
    cache.set("relevance", question, answer, score)
    return score


def context_precision(llm: LLM, question: str, docs: list[dict]) -> float:
    if not docs:
        return 0.0
    hits = sum(
        1
        for d in docs
        if "yes"
        in llm.ask(
            _CTX_PRECISION_PROMPT.format(question=question, passage=d["text"][:600])
        ).lower()
    )
    return hits / len(docs)


def rouge_l(answer: str, reference: str) -> float:
    scores = _scorer.score(reference, answer)
    return round(scores["rougeL"].fmeasure, 4)


def _parse_score(raw: str) -> float:
    try:
        val = float(raw.strip().split()[0])
        return max(0.0, min(1.0, val))
    except (ValueError, IndexError):
        return 0.5
