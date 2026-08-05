"""
Correlation analysis of graph metrics vs. accuracy.

Usage:
    python scripts/analyze.py results/results.json
    python scripts/analyze.py results/results.json --k 3 5
    python scripts/analyze.py results/run_X.json --stratify-by-size
    python scripts/analyze.py results/run_X.json --target acc_per_token

Outputs a table: metric → ρ, ρ|n (partial rank corr controlling n_agents), τ,
regret@k, NDCG@k, n, total_tokens
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._common import n_agents_map


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


def _partial_spearman(x: List[float], y: List[float], z: List[float]) -> float:
    """Spearman rho between x and y after removing the linear rank-effect of z
    from both (i.e. Pearson corr of the rank-residuals)."""
    import numpy as np
    rx, ry, rz = _ranks(x), _ranks(y), _ranks(z)

    def resid(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        A = np.vstack([np.ones_like(b), b]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef

    ex, ey = resid(rx, rz), resid(ry, rz)
    if ex.std() < 1e-8 * max(np.std(rx), 1e-12) or ey.std() < 1e-8 * max(np.std(ry), 1e-12):
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


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


def stratified_metric(target_vals: List[float], metric_vals: List[float], n_agents_vals: List[int], k: int):
    """regret@k / NDCG@k computed within strata of equal n_agents, averaged
    across strata weighted by stratum size. Strata smaller than k are
    skipped (returned separately). Returns (None, None, skipped) if every
    stratum was skipped."""
    strata = {}
    for i, na in enumerate(n_agents_vals):
        strata.setdefault(na, []).append(i)

    total_w = 0
    reg_sum = nd_sum = 0.0
    skipped = []
    for na, idxs in strata.items():
        if len(idxs) < k:
            skipped.append(na)
            continue
        sub_t = [target_vals[i] for i in idxs]
        sub_m = [metric_vals[i] for i in idxs]
        order = sorted(range(len(idxs)), key=lambda j: sub_m[j], reverse=True)
        w = len(idxs)
        reg_sum += regret_at_k(sub_t, order, k) * w
        nd_sum += ndcg_at_k(sub_t, order, k) * w
        total_w += w

    if total_w == 0:
        return None, None, skipped
    return reg_sum / total_w, nd_sum / total_w, skipped


def analyze(results: list, ks: List[int], target: str = "accuracy", stratify_by_size: bool = False) -> None:
    accs = [r["accuracy"] for r in results]
    n = len(accs)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    topo_names = [r["topology"] for r in results]

    agents_map = n_agents_map()
    n_agents_vals = []
    for name in topo_names:
        if name not in agents_map:
            raise KeyError(name)
        n_agents_vals.append(agents_map[name])

    if target == "accuracy":
        target_vals = accs
    elif target == "acc_per_token":
        target_vals = [
            (a / (r.get("total_tokens", 0) / 1e6)) if r.get("total_tokens", 0) > 0 else 0.0
            for a, r in zip(accs, results)
        ]
    else:
        raise ValueError(f"unknown target: {target}")

    metric_keys = list(results[0]["metrics"].keys())

    print(f"Target: {target}\n")

    header_parts = ["metric".ljust(24)] + ["ρ".rjust(7), "ρ|n".rjust(7), "τ".rjust(7)]
    for k in ks:
        header_parts += [f"reg@{k}".rjust(8), f"NDCG@{k}".rjust(8)]
    header_parts += ["n".rjust(4), "total_tokens".rjust(12)]
    print("  ".join(header_parts))
    print("-" * (24 + 10 * (3 + 2 * len(ks)) + 6 + 14))

    def _print_row(label: str, metric_vals: List[float]) -> None:
        rho = _spearman(metric_vals, target_vals)
        rho_partial = _partial_spearman(metric_vals, target_vals, n_agents_vals)
        tau = _kendall(metric_vals, target_vals)
        row = [label.ljust(24), f"{rho:+.4f}".rjust(7), f"{rho_partial:+.4f}".rjust(7), f"{tau:+.4f}".rjust(7)]
        desc_order = sorted(range(n), key=lambda i: metric_vals[i], reverse=True)
        for k in ks:
            reg = regret_at_k(target_vals, desc_order, k)
            nd = ndcg_at_k(target_vals, desc_order, k)
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

    n_agents_float = [float(c) for c in n_agents_vals]
    _print_row("baseline:n_agents", n_agents_float)

    print(f"\nn={n}  total_tokens={total_tokens}")

    if stratify_by_size:
        print("\nStratified by size (--stratify-by-size): regret@k / NDCG@k computed within")
        print("n_agents strata and averaged across strata weighted by stratum size.")
        for mk in metric_keys + ["baseline:random", "baseline:n_agents"]:
            if mk == "baseline:random":
                vals = rand_vals
            elif mk == "baseline:n_agents":
                vals = n_agents_float
            else:
                vals = [r["metrics"][mk] for r in results]
            row_bits = [mk.ljust(24)]
            skipped_all = set()
            for k in ks:
                reg, nd, skipped = stratified_metric(target_vals, vals, n_agents_vals, k)
                skipped_all.update(skipped)
                cell = f"reg@{k}=n/a NDCG@{k}=n/a" if reg is None else f"reg@{k}={reg:.4f} NDCG@{k}={nd:.4f}"
                row_bits.append(cell.rjust(28))
            print("  ".join(row_bits))
            if skipped_all:
                print(f"    (strata with n_agents in {sorted(skipped_all)} skipped: smaller than k)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file", nargs="?", default="results/results.json")
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--stratify-by-size", action="store_true",
                         help="compute regret@k/NDCG@k within n_agents strata (does this metric rank same-size topologies?)")
    parser.add_argument("--target", choices=["accuracy", "acc_per_token"], default="accuracy",
                         help="ranking target: raw accuracy, or accuracy per 1M tokens")
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
    analyze(results, args.k, target=args.target, stratify_by_size=args.stratify_by_size)


if __name__ == "__main__":
    main()
