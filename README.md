# Topology Estimator — PoC

Proof-of-concept: do **graph-structural metrics of a MAS topology correlate with its accuracy** on math reasoning tasks — without prompt tuning or training? Research proposal available at [mas_surrogate.pdf](./mas_surrogate.pdf) (in Russian).

## Hypothesis

Given a multi-agent system where agents communicate along a directed graph, topology alone (diameter, spectral gap, degree distribution, etc.) should predict task accuracy well enough to rank topologies before running any LLM calls.

**Current status:** ❌ Not confirmed at N=20. Max |Spearman r| = 0.16. See [poc_results.md](./poc_results.md).

## Repository structure

```
topology_estimator/
├── topologies/
│   └── definitions.py      # 18 topologies as nx.DiGraph
├── mas/
│   ├── agent.py            # Agent class — stub mode + real OpenAI calls
│   ├── runner.py           # MASRunner — topological-sort execution
│   └── prompts.py          # System prompts, role assignment, answer parser
├── metrics/
│   └── graph_metrics.py    # TopologyMetrics: 7 structural metrics
├── scripts/
│   └── run_poc.py          # CLI entrypoint — runs experiment, saves results/
├── notebook/
│   ├── run_poc.ipynb       # Interactive version of run_poc.py
│   └── analysis.ipynb      # Scatter plots, correlations, MI, top-K, radar
├── results/                # Auto-generated: results.json, run_*.json, *.png
├── poc_results.md          # Human-readable experiment summaries
└── pyproject.toml
```

## Topologies (18)

**Parametric** — n ∈ {3, 5}:

| Family | Structure |
|--------|-----------|
| `chain_n` | `task → A0 → A1 → … → A{n-1}` |
| `star_n` | `task → A0, A1, …, A{n-1}` (parallel, no inter-agent edges) |
| `fc_n` | `task → all`, `Ai → Aj` for all i < j (fully connected DAG) |
| `hierarchical_n` | Two-level hierarchy with a manager node |
| `two_layer_ensemble_n` | Parallel solvers → aggregator layer |

**Fixed-schema hybrids:**

| Name | Structure |
|------|-----------|
| `debate` | Two agents exchange solutions before aggregator |
| `pipeline_with_critic` | `task → solver → critic → aggregator` |
| `star_then_chain` | Star fan-out → chain continuation |
| `chain_with_star_sink` | Chain backbone + all nodes feed a shared sink |
| `fc_then_sink` | FC first layer → single aggregator |
| `star_with_chain_backbone` | Chain + skip-connections from task to each node |
| `two_stars_merged` | Two parallel star branches → shared aggregator |
| `chain_of_stars` | Fan-out → fan-in → fan-out (hourglass) |

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
python scripts/run_poc.py --stub --n-questions 3
```

**Real API — GSM8K (grade-school arithmetic)**:
```bash
python scripts/run_poc.py --model gpt-3.5-turbo --dataset gsm8k --n-questions 20
```

**Real API — MATH-500 (harder, competition math)**:
```bash
python scripts/run_poc.py --model gpt-4o-mini --dataset math500 --n-questions 20
```

**All 18 topologies** (adds `--all-topologies`):
```bash
python scripts/run_poc.py --model gpt-4o-mini --dataset math500 --n-questions 50 --all-topologies
```

Each run saves `results/run_<timestamp>.json` and updates `results/runs_index.json`.

**Analysis** — after results are generated, open `notebook/analysis.ipynb`.

## Graph metrics (7)

| Metric | Description |
|--------|-------------|
| `diameter` | Average shortest path length (undirected) |
| `avg_degree` | Mean out-degree across all nodes |
| `structural_entropy` | Shannon entropy of degree distribution |
| `spectral_gap` | λ₂ of normalised Laplacian (algebraic connectivity) |
| `task_centrality` | Betweenness centrality of the virtual `task` input node |
| `flow_entropy` | Stationary-distribution–weighted transition entropy |
| `von_neumann_entropy` | Spectral entropy of the normalised Laplacian |

## Agent roles

| Role | Behaviour |
|------|-----------|
| `solver` | Produces an independent numerical answer |
| `critic` | Reviews predecessors' solutions, gives corrected answer |
| `aggregator` | Picks the most likely correct answer from all inputs |
| `manager` | Outlines solution strategy for the team |
| `debater` | Solves independently, ignores other agents |
| `judge` | Evaluates multiple solutions, selects the best |

Roles are assigned by topological position (first = solver, middle = critic, last = aggregator) or via `HYBRID_ROLES` map for fixed-schema topologies. Each agent ends its response with `ANSWER: <number>`; the runner extracts the final node's answer.

## Datasets

| Dataset | Source | Difficulty | Answer format |
|---------|--------|------------|---------------|
| `gsm8k` | `openai/gsm8k` on HuggingFace | Grade-school arithmetic | `#### <int>` |
| `math500` | `HuggingFaceH4/MATH-500` | Competition math | `\boxed{<expr>}` |

## Key findings (as of 2026-05-24)

| Run | Model | Dataset | N | Topologies | Mean acc | Best topology | Max \|Spearman r\| |
|-----|-------|---------|---|------------|----------|--------------|-------------------|
| `20260524_075933` | gpt-3.5-turbo | GSM8K | 20 | 18 | 0.242 | pipeline_with_critic (0.40) | 0.16 |
| `20260524_050322` | gpt-4o-mini | math500 | 20 | 18 | 0.456 | fc_then_sink / debate (0.55) | — |

Neither run shows meaningful metric–accuracy correlation. Ranking of topologies inverts between models. N=20 is insufficient — 95% CI ≈ ±11 pp while the full accuracy range is 15 pp.

## Output

Each run produces:
- `results/run_<id>.json` — per-topology accuracy + metrics + metadata
- `results/runs_index.json` — index of all runs
- `results/scatter_plots.png` — metric vs accuracy scatter (7 subplots)
- `results/radar.png` — normalised metric profiles per topology
- `results/mutual_info.png` — MI(metric, accuracy) bar chart
- `results/topk_accuracy.png` — top-K proxy accuracy vs random baseline
