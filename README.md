# Projet-RAG

Laboratory to compare **7 RAG retrieval strategies** on the same corpus with shared embeddings, LLM, and benchmark metrics.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_install.py
cp .env.example .env
# Edit .env with your API keys
```

## Environment variables

See [`.env.example`](.env.example). Main options:

| Variable | Description |
|----------|-------------|
| `RAG_PROVIDER` | `ollama_cloud` (default), `ollama_local`, `openai` |
| `OLLAMA_API_KEY` | Ollama Cloud API key |
| `RAG_MODEL` | Generation model (default: `gpt-oss:120b`) |
| `EVAL_MODEL` | Model for benchmark judge metrics (defaults to `RAG_MODEL`) |
| `COMPRESSION_MODE` | `llm` or `extractive` (default: `extractive`) |

## CLI

```bash
# Index documents (incremental by default; use --fresh to reset)
python main.py index data/samples/ -s recursive
python main.py index data/samples/ --fresh -e naive

# Single question
python main.py ask "How does HyDE work?" -e hyde

# Benchmark (index first)
python main.py bench data/questions.txt -o results/benchmark.json

# Index all experiments + full benchmark
python main.py run-all data/questions.txt data/samples/
```

## Experiments

| Name | Description |
|------|-------------|
| `naive` | Embed query → top-k → generate |
| `hyde` | Hypothetical document embedding |
| `fusion` | Multi-query + RRF |
| `reranker` | Bi-encoder fetch + cross-encoder rerank |
| `hybrid` | Dense + BM25 + RRF |
| `parent_child` | Small chunks for search, parent for context |
| `compression` | Extractive or LLM compression before generation |

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

## Ground truth chunk IDs

After changing chunking or adding documents, regenerate `relevant_chunk_ids` in `data/ground_truth.json`:

```bash
python scripts/list_chunk_ids.py data/samples/
```

Match chunk text to your ground-truth entries manually or with the script output.

## Project layout

```
src/           # loader, chunker, embedder, vectorstore, bm25, llm, config
experiments/   # 7 RAG implementations
evaluation/    # benchmark + metrics
data/samples/  # corpus
data/questions.txt
data/ground_truth.json
```
