"""
Correlation analysis of graph metrics vs. accuracy.

Usage:
    python scripts/analyze.py results/results.json
    python scripts/analyze.py results/results.json --k 3 5

Outputs a table: metric → ρ, τ, regret@k, NDCG@k, n, total_tokens
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))


def _spearman(x: List[float], y: List[float]) -> float:
    n = len(x)
    rx = _ranks(x)
    ry = _ranks(y)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def _kendall(x: List[float], y: List[float]) -> float:
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = x[i] - x[j]
            sy = y[i] - y[j]
            if sx * sy > 0:
                concordant += 1
            elif sx * sy < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom else 0.0


def _ranks(x: List[float]) -> List[float]:
    sorted_idx = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j < len(x) - 1 and x[sorted_idx[j]] == x[sorted_idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _dcg(relevances: List[float], k: int) -> float:
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(true_scores: List[float], pred_order: List[int], k: int) -> float:
    """pred_order: indices sorted by surrogate metric descending."""
    predicted_rels = [true_scores[i] for i in pred_order]
    ideal_rels = sorted(true_scores, reverse=True)
    dcg = _dcg(predicted_rels, k)
    idcg = _dcg(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def regret_at_k(true_scores: List[float], pred_order: List[int], k: int) -> float:
    """true_best_accuracy − best_accuracy_in_surrogate_top_k."""
    best = max(true_scores)
    top_k_scores = [true_scores[i] for i in pred_order[:k]]
    return best - max(top_k_scores)


def analyze(results: list, ks: List[int]) -> None:
    accs = [r["accuracy"] for r in results]
    n = len(accs)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    metric_keys = list(results[0]["metrics"].keys())

    header_parts = ["metric".ljust(24)] + ["ρ".rjust(7)] + ["τ".rjust(7)]
    for k in ks:
        header_parts += [f"reg@{k}".rjust(8), f"NDCG@{k}".rjust(8)]
    header_parts += ["n".rjust(4), "total_tokens".rjust(12)]
    print("  ".join(header_parts))
    print("-" * (24 + 10 * (2 + 2 * len(ks)) + 6 + 14))

    def _print_row(label: str, metric_vals: List[float]) -> None:
        rho = _spearman(metric_vals, accs)
        tau = _kendall(metric_vals, accs)
        row = [label.ljust(24), f"{rho:+.4f}".rjust(7), f"{tau:+.4f}".rjust(7)]
        desc_order = sorted(range(n), key=lambda i: metric_vals[i], reverse=True)
        for k in ks:
            reg = regret_at_k(accs, desc_order, k)
            nd = ndcg_at_k(accs, desc_order, k)
            row += [f"{reg:.4f}".rjust(8), f"{nd:.4f}".rjust(8)]
        row.append(str(n).rjust(4))
        row.append(str(total_tokens).rjust(12))
        print("  ".join(row))

    for mk in metric_keys:
        vals = [r["metrics"][mk] for r in results]
        _print_row(mk, vals)

    # ── baselines ─────────────────────────────────────────────────────────
    import random
    rng = random.Random(0)
    shuffled = list(range(n))
    rng.shuffle(shuffled)
    rand_vals = [shuffled[i] for i in range(n)]
    _print_row("baseline:random", rand_vals)

    agent_counts = []
    for r in results:
        import networkx as nx
        # agent count = number of non-task nodes (approximated by topology name if graph unavailable)
        # load graph from definitions
        agent_counts.append(r.get("_n_agents", len(r["metrics"])))

    # better: load actual graphs
    try:
        from topologies.definitions import get_all_topologies
        all_topos = get_all_topologies()
        agent_counts = [
            sum(1 for node in all_topos[r["topology"]].nodes if node != "task")
            if r["topology"] in all_topos else 0
            for r in results
        ]
    except Exception:
        pass

    _print_row("baseline:n_agents", [float(c) for c in agent_counts])

    print(f"\nn={n}  total_tokens={total_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", nargs="?", default="results/results.json")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5])
    args = parser.parse_args()

    path = Path(args.results_file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    doc = json.loads(path.read_text())
    # support both flat list and run_doc format
    results = doc if isinstance(doc, list) else doc.get("results", doc)

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(results)} topologies from {path}\n")
    analyze(results, args.k)


if __name__ == "__main__":
    main()
