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
    latency: float
    num_docs: int
    queries_used: int


def run_benchmark(
    experiments: list[BaseRAG],
    questions: list[str],
    llm: LLM,
    output_path: str = "./results/benchmark.json",
    ground_truth: dict[str, str] | None = None,
) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing checkpoint to skip already-completed (experiment, question) pairs
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
            rl = rouge_l(result.answer, ground_truth[q]) if ground_truth and q in ground_truth else 0.0

            row = asdict(BenchmarkRow(
                experiment=exp.name,
                question=q,
                answer=result.answer,
                faithfulness=faith,
                relevance=rel,
                context_precision=ctx,
                rouge_l=rl,
                latency=result.latency,
                num_docs=len(result.retrieved_docs),
                queries_used=len(result.queries_used),
            ))
            rows.append(row)
            # Checkpoint after every row
            output_path.write_text(json.dumps(rows, indent=2))

    df = pd.DataFrame(rows)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
    _print_summary(df, has_rouge=ground_truth is not None)
    return df


def _print_summary(df: pd.DataFrame, has_rouge: bool = False) -> None:
    agg = {
        "faithfulness": ("faithfulness", "mean"),
        "relevance": ("relevance", "mean"),
        "context_precision": ("context_precision", "mean"),
        "latency": ("latency", "mean"),
        "queries_used": ("queries_used", "mean"),
    }
    if has_rouge:
        agg["rouge_l"] = ("rouge_l", "mean")

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
        cols += [f"{row['latency']:.1f}", f"{row['queries_used']:.1f}"]
        table.add_row(*cols)

    console.print("\n")
    console.print(table)


def _color(val: float) -> str:
    color = "green" if val >= 0.7 else ("yellow" if val >= 0.4 else "red")
    return f"[{color}]{val:.3f}[/{color}]"
