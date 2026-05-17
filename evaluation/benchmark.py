import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table

from evaluation.metrics import (
    answer_relevance,
    context_precision,
    faithfulness,
    rouge_l,
)
from evaluation.retrieval import extract_retrieved_ids, mrr, recall_at_k
from experiments.base import BaseRAG
from src.llm import LLM

console = Console()


@dataclass
class BenchmarkRow:
    experiment: str
    question: str
    answer: str
    faithfulness: float
    relevance: float
    context_precision: float
    rouge_l: float
    recall_at_5: float
    mrr: float
    latency: float
    num_docs: int
    queries_used: int


def run_benchmark(
    experiments: list[BaseRAG],
    questions: list[str],
    llm: LLM,
    output_path: str = "./results/benchmark.json",
    ground_truth: dict[str, dict] | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    rows: list[dict] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            rows = existing
            done = {(r["experiment"], r["question"]) for r in existing}
            if done:
                console.print(f"[dim]Resuming — {len(done)} rows already done[/dim]")
        except Exception:
            pass

    for exp in experiments:
        pending = [q for q in questions if (exp.name, q) not in done]
        if not pending:
            console.print(f"[dim]── {exp.name}: already complete, skipping[/dim]")
            continue

        console.print(f"\n[bold cyan]── {exp.name} ({len(pending)}/{len(questions)} remaining) ──[/bold cyan]")
        for q in track(pending, description=f"  {exp.name}"):
            result = exp.query(q)
            faith = faithfulness(llm, q, result)
            rel = answer_relevance(llm, q, result.answer)
            ctx = context_precision(llm, q, result.retrieved_docs)

            gt_entry = ground_truth.get(q, {}) if ground_truth else {}
            reference = gt_entry.get("reference", "")
            rl = rouge_l(result.answer, reference) if reference else 0.0

            relevant_ids = gt_entry.get("relevant_chunk_ids", [])
            retrieved_ids = extract_retrieved_ids(result.retrieved_docs)
            rec5 = recall_at_k(retrieved_ids, relevant_ids, k=5) if relevant_ids else 0.0
            mrr_score = mrr(retrieved_ids, relevant_ids) if relevant_ids else 0.0

            row = asdict(BenchmarkRow(
                experiment=exp.name,
                question=q,
                answer=result.answer,
                faithfulness=faith,
                relevance=rel,
                context_precision=ctx,
                rouge_l=rl,
                recall_at_5=rec5,
                mrr=mrr_score,
                latency=result.latency,
                num_docs=len(result.retrieved_docs),
                queries_used=len(result.queries_used),
            ))
            rows.append(row)
            output_path.write_text(json.dumps(rows, indent=2))

    df = pd.DataFrame(rows)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
    has_rouge = bool(ground_truth)
    has_retrieval = bool(
        ground_truth
        and any(gt.get("relevant_chunk_ids") for gt in ground_truth.values())
    )
    _print_summary(df, has_rouge=has_rouge, has_retrieval=has_retrieval)
    return df


def _print_summary(
    df: pd.DataFrame,
    has_rouge: bool = False,
    has_retrieval: bool = False,
) -> None:
    agg = {
        "faithfulness": ("faithfulness", "mean"),
        "relevance": ("relevance", "mean"),
        "context_precision": ("context_precision", "mean"),
        "latency": ("latency", "mean"),
        "queries_used": ("queries_used", "mean"),
    }
    if has_rouge:
        agg["rouge_l"] = ("rouge_l", "mean")
    if has_retrieval:
        agg["recall_at_5"] = ("recall_at_5", "mean")
        agg["mrr"] = ("mrr", "mean")

    summary = (
        df.groupby("experiment")
        .agg(**agg)
        .round(3)
        .sort_values("relevance", ascending=False)
    )

    table = Table(title="RAG Benchmark — Summary", show_lines=True, highlight=True)
    table.add_column("Method", style="bold white")
    table.add_column("Faithfulness ↑", justify="center")
    table.add_column("Relevance ↑", justify="center")
    table.add_column("Ctx Precision ↑", justify="center")
    if has_rouge:
        table.add_column("ROUGE-L ↑", justify="center")
    if has_retrieval:
        table.add_column("Recall@5 ↑", justify="center")
        table.add_column("MRR ↑", justify="center")
    table.add_column("Latency (s) ↓", justify="center")
    table.add_column("Queries", justify="center")

    for name, row in summary.iterrows():
        cols = [
            str(name),
            _color(row["faithfulness"]),
            _color(row["relevance"]),
            _color(row["context_precision"]),
        ]
        if has_rouge:
            cols.append(_color(row["rouge_l"]))
        if has_retrieval:
            cols.append(_color(row["recall_at_5"]))
            cols.append(_color(row["mrr"]))
        cols += [f"{row['latency']:.1f}", f"{row['queries_used']:.1f}"]
        table.add_row(*cols)

    console.print("\n")
    console.print(table)


def _color(val: float) -> str:
    color = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
    return f"[{color}]{val:.3f}[/{color}]"
