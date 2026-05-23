# Topology Estimator — PoC

Proof-of-concept: do **graph-structural metrics of a MAS topology correlate with its accuracy** on math reasoning tasks — without prompt tuning or training?

## Hypothesis

Given a multi-agent system where agents communicate along a directed graph, topology alone (diameter, spectral gap, degree distribution, etc.) should predict task accuracy well enough to rank topologies before running any LLM calls.

## Repository structure

```
topology_estimator/
├── topologies/
│   └── definitions.py      # 4 topologies as nx.DiGraph: chain, star, full, random
├── mas/
│   ├── agent.py            # Agent class with stub mode (no API) and real OpenAI calls
│   ├── runner.py           # MASRunner — topological-sort execution of the graph
│   └── prompts.py          # System prompts + role assignment + answer parser
├── metrics/
│   └── graph_metrics.py    # TopologyMetrics: diameter, avg_degree, entropy, spectral_gap, centrality
├── scripts/
│   └── run_poc.py          # CLI entrypoint — runs full experiment, saves results.json
├── notebook/
│   ├── run_poc.ipynb       # Interactive version of run_poc.py, cell-by-cell
│   └── analysis.ipynb      # Scatter plots, Spearman correlations, radar chart, ranking
├── results/                # Auto-generated: results.json, *.png
├── data/                   # GSM8K samples (downloaded at runtime)
├── .env                    # OPENAI_API_KEY (not committed)
└── pyproject.toml
```

## Topologies

| Name   | Structure                                              |
|--------|--------------------------------------------------------|
| chain  | `task → A0 → A1 → A2`                                 |
| star   | `task → A0, A1, A2 → A3 (aggregator)`                 |
| full   | Full DAG: task → all, Ai → Aj for i < j               |
| random | Random DAG, seed=42, guaranteed reachability from task |

## Setup

```bash
conda activate langgraph_env
pip install -e .
```

Create `.env` in the project root:

```
OPENAI_API_KEY=sk-...
```

## Running

**Stub mode** (no API calls, pipeline smoke-test):
```bash
python scripts/run_poc.py --stub --n-questions 5
```

**Real API**:
```bash
python scripts/run_poc.py --model gpt-3.5-turbo --n-questions 20
```

**Interactive** — open `notebook/run_poc.ipynb` in Jupyter with kernel `langgraph_env`.  
Run the dotenv cell first, then step through cells one by one.

**Analysis** — after `results/results.json` is generated, open `notebook/analysis.ipynb`.

## Graph metrics

| Metric               | Description                                              |
|----------------------|----------------------------------------------------------|
| `diameter`           | Average shortest path length (undirected)                |
| `avg_degree`         | Mean out-degree across all nodes                         |
| `structural_entropy` | Shannon entropy of degree distribution                   |
| `spectral_gap`       | λ₂ of normalised Laplacian (algebraic connectivity)      |
| `task_centrality`    | Betweenness centrality of the virtual `task` input node  |

## Output

`results/results.json`:
```json
[
  {"topology": "chain",  "accuracy": 0.55, "metrics": {"diameter": 2.5, ...}},
  {"topology": "star",   "accuracy": 0.60, "metrics": {...}},
  {"topology": "full",   "accuracy": 0.65, "metrics": {...}},
  {"topology": "random", "accuracy": 0.50, "metrics": {...}}
]
```

`results/` also contains `scatter_plots.png`, `radar.png`, `topologies.png`.

## Dataset

GSM8K (grade-school math) — loaded via HuggingFace `openai/gsm8k`, falls back to raw GitHub download if unavailable. Ground-truth answers parsed with `r"####\s*(-?\d+)"`.

## Agent roles

Roles are assigned cyclically: `solver → critic → aggregator`.  
Each agent appends `ANSWER: <number>` at the end of its response; the runner extracts the final agent's answer as the system output.
