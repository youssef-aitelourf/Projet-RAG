"""
RAG Experimentation CLI — Ollama Cloud / gpt-oss:120b

Commands:
  index    <path>                    Index docs into all (or selected) experiments
  ask      <question>                Quick single query
  bench    <questions.txt>           Full benchmark (experiments must be indexed first)
  run-all  <questions.txt> <path>    Index + benchmark in one shot
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import click
from rich.console import Console
from rich.panel import Panel

from src.config import Config
from src.loader import load_dir, load_file
from src.llm import LLM
from experiments.naive import NaiveRAG
from experiments.hyde import HyDERAG
from experiments.fusion import RAGFusion
from experiments.reranker import RerankerRAG
from experiments.hybrid import HybridRAG
from experiments.parent_child import ParentChildRAG
from experiments.compression import CompressionRAG
from evaluation.benchmark import run_benchmark

console = Console()

EXPERIMENTS: dict[str, type] = {
    "naive":        NaiveRAG,
    "hyde":         HyDERAG,
    "fusion":       RAGFusion,
    "reranker":     RerankerRAG,
    "hybrid":       HybridRAG,
    "parent_child": ParentChildRAG,
    "compression":  CompressionRAG,
}


def _load_ground_truth(path: str = "./data/ground_truth.json") -> dict[str, dict] | None:
    p = Path(path)
    if not p.exists():
        return None
    items = json.loads(p.read_text())
    return {item["question"]: item for item in items}


@click.group()
def cli():
    """RAG experiment suite — Ollama Cloud / gpt-oss:120b"""


@cli.command()
@click.argument("path")
@click.option("--strategy", "-s", default="recursive",
              type=click.Choice(["fixed", "recursive", "sentence"]))
@click.option("--experiment", "-e", multiple=True, default=list(EXPERIMENTS.keys()),
              type=click.Choice(list(EXPERIMENTS.keys())))
def index(path: str, strategy: str, experiment: tuple[str, ...]):
    """Index documents from a file or directory."""
    cfg = Config()
    p = Path(path)
    chunks = load_file(p, strategy=strategy) if p.is_file() else load_dir(p, strategy=strategy)
    console.print(Panel(f"[green]{len(chunks)} chunks[/green] from [bold]{path}[/bold]"))
    for name in experiment:
        exp = EXPERIMENTS[name](cfg, collection=name)
        exp.index(chunks)
        console.print(f"  [green]✓[/green] {name}")


@cli.command()
@click.argument("question")
@click.option("--experiment", "-e", default="naive",
              type=click.Choice(list(EXPERIMENTS.keys())))
def ask(question: str, experiment: str):
    """Ask a single question with one RAG method."""
    cfg = Config()
    exp = EXPERIMENTS[experiment](cfg, collection=experiment)
    result = exp.query(question)
    console.print(Panel(result.answer, title=f"[bold]{experiment}[/bold]", expand=False))
    console.print(f"[dim]{result.latency:.2f}s | {len(result.retrieved_docs)} docs | queries: {result.queries_used}[/dim]")
    console.print("\n[bold]Retrieved passages:[/bold]")
    for i, doc in enumerate(result.retrieved_docs, 1):
        console.print(f"  [{i}] score={doc['score']:.3f} | {doc['text'][:120]}...")


@cli.command()
@click.argument("questions_file")
@click.option("--experiment", "-e", multiple=True, default=list(EXPERIMENTS.keys()),
              type=click.Choice(list(EXPERIMENTS.keys())))
@click.option("--output", "-o", default="./results/benchmark_v2.json")
@click.option("--gt", default="./data/ground_truth.json", help="Ground truth JSON path")
def bench(questions_file: str, experiment: tuple[str, ...], output: str, gt: str):
    """Run benchmark (experiments must be indexed first with `index`)."""
    cfg = Config()
    questions = [
        q.strip() for q in Path(questions_file).read_text().splitlines()
        if q.strip() and not q.startswith("#")
    ]
    ground_truth = _load_ground_truth(gt)
    exps = [EXPERIMENTS[n](cfg, collection=n) for n in experiment]
    llm = LLM(cfg, role="eval")
    run_benchmark(exps, questions, llm, output_path=output, ground_truth=ground_truth)


@cli.command()
@click.argument("questions_file")
@click.argument("path")
@click.option("--strategy", "-s", default="recursive",
              type=click.Choice(["fixed", "recursive", "sentence"]))
@click.option("--output", "-o", default="./results/benchmark_v2.json")
@click.option("--gt", default="./data/ground_truth.json")
def run_all(questions_file: str, path: str, strategy: str, output: str, gt: str):
    """Index docs and run full benchmark in one shot."""
    cfg = Config()
    p = Path(path)
    chunks = load_file(p, strategy=strategy) if p.is_file() else load_dir(p, strategy=strategy)
    console.print(f"[green]{len(chunks)} chunks loaded[/green]")

    exps = []
    for name, cls in EXPERIMENTS.items():
        exp = cls(cfg, collection=name)
        exp.index(chunks)
        console.print(f"  [green]✓[/green] {name} indexed")
        exps.append(exp)

    questions = [
        q.strip() for q in Path(questions_file).read_text().splitlines()
        if q.strip() and not q.startswith("#")
    ]
    ground_truth = _load_ground_truth(gt)
    llm = LLM(cfg, role="eval")
    run_benchmark(exps, questions, llm, output_path=output, ground_truth=ground_truth)


if __name__ == "__main__":
    cli()
