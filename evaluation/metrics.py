from rouge_score import rouge_scorer

from src.llm import LLM
from experiments.base import RAGResult

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
    raw = llm.ask(
        _FAITHFULNESS_PROMPT.format(
            context=context[:3000], question=question, answer=result.answer
        )
    )
    return _parse_score(raw)


def answer_relevance(llm: LLM, question: str, answer: str) -> float:
    raw = llm.ask(_RELEVANCE_PROMPT.format(question=question, answer=answer))
    return _parse_score(raw)


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
