"""
statsmodels-based econometric evaluation of topology structural metrics vs.
accuracy for one run. Replicates notebook/analysis.ipynb's correlation
table/plots and adds two tiers of proper hypothesis testing statsmodels
enables that a spreadsheet of one-at-a-time correlations can't:

  Tier 1 (topology-level, n = n_topologies): multivariate OLS across all 5
  metrics at once (not one-at-a-time), HC3 robust SEs, VIF multicollinearity
  check, Benjamini-Hochberg correction across the 5 univariate p-values.

  Tier 2 (question-level panel, n = n_topologies * n_questions): logistic
  regression of match~metrics with cluster-robust SEs (clustered by
  topology, since the metrics are constant within a topology — naive SEs
  would understate uncertainty via pseudo-replication), plus a
  question-fixed-effects variant that absorbs per-question difficulty.

Usage:
    python scripts/stats_analysis.py <run_id>
    python scripts/stats_analysis.py results/20260725_101948
    python scripts/stats_analysis.py results/run_20260725_101948.json

Writes results/<run_id>/analysis/{topology_level.txt, question_level.txt,
panel_data.csv, scatter_plots.png, mutual_info.png, topk_accuracy.png,
radar.png}.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent.parent
METRIC_COLS = ["diameter", "avg_degree", "structural_entropy", "spectral_gap", "task_centrality"]


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(arg: str) -> tuple:
    """Accepts a bare run_id, a log-folder path, or a run_<id>.json path."""
    out_dir = ROOT / "results"
    p = Path(arg)
    if p.suffix == ".json":
        run_id = p.stem.removeprefix("run_")
    else:
        run_id = p.name if p.is_absolute() or p.exists() else arg
    run_file = out_dir / f"run_{run_id}.json"
    log_dir = out_dir / run_id
    if not run_file.exists():
        raise FileNotFoundError(f"aggregate run file not found: {run_file}")
    if not log_dir.exists():
        raise FileNotFoundError(f"per-topology log folder not found: {log_dir}")
    return run_id, run_file, log_dir


def load_topo_df(run_file: Path) -> pd.DataFrame:
    doc = json.loads(run_file.read_text())
    rows = []
    for r in doc["results"]:
        row = {"topology": r["topology"], "accuracy": r["accuracy"], "total_tokens": r.get("total_tokens", 0)}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_panel_df(log_dir: Path, topo_df: pd.DataFrame) -> pd.DataFrame:
    metric_lookup = topo_df.set_index("topology")[METRIC_COLS].to_dict("index")
    rows = []
    for log_path in sorted(log_dir.glob("*.json")):
        topo_name = log_path.stem
        if topo_name not in metric_lookup:
            continue
        doc = json.loads(log_path.read_text())
        for rec in doc["records"]:
            row = {
                "topology": topo_name, "qi": rec["qi"],
                "match": int(bool(rec["match"])), "tokens": rec.get("tokens", 0),
            }
            row.update(metric_lookup[topo_name])
            rows.append(row)
    return pd.DataFrame(rows)


def _active_metrics(df: pd.DataFrame) -> tuple:
    """Drops zero-variance metric columns (would blow up OLS/VIF)."""
    keep = [c for c in METRIC_COLS if df[c].std(ddof=0) > 0]
    dropped = [c for c in METRIC_COLS if c not in keep]
    return keep, dropped


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — topology-level
# ─────────────────────────────────────────────────────────────────────────────

def correlation_table(topo_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRIC_COLS:
        x, y = topo_df[m].values, topo_df["accuracy"].values
        if np.std(x) > 0:
            r, rp = stats.pearsonr(x, y)
            rho, sp = stats.spearmanr(x, y)
            tau, tp = stats.kendalltau(x, y)
        else:
            r = rho = tau = 0.0
            rp = sp = tp = 1.0
        rows.append({
            "metric": m, "pearson_r": r, "pearson_p": rp,
            "spearman_r": rho, "spearman_p": sp, "kendall_tau": tau, "kendall_p": tp,
        })
    df = pd.DataFrame(rows)
    df["pearson_q_bh"] = multipletests(df["pearson_p"], method="fdr_bh")[1]
    df["spearman_q_bh"] = multipletests(df["spearman_p"], method="fdr_bh")[1]
    return df


def multivariate_ols(topo_df: pd.DataFrame, standardized: bool):
    keep, dropped = _active_metrics(topo_df)
    X = topo_df[keep].copy()
    if standardized:
        X = (X - X.mean()) / X.std(ddof=0)
    X = sm.add_constant(X)
    model = sm.OLS(topo_df["accuracy"], X).fit(cov_type="HC3")
    return model, dropped


def vif_table(topo_df: pd.DataFrame) -> pd.DataFrame:
    keep, _ = _active_metrics(topo_df)
    X = sm.add_constant(topo_df[keep])
    rows = [
        {"metric": col, "VIF": variance_inflation_factor(X.values, i)}
        for i, col in enumerate(X.columns) if col != "const"
    ]
    return pd.DataFrame(rows)


def run_tier1(topo_df: pd.DataFrame, out_dir: Path) -> list:
    lines = [f"Tier 1 — topology-level (n={len(topo_df)} topologies)", "=" * 70, ""]

    corr = correlation_table(topo_df)
    lines += ["Univariate correlation vs. accuracy (BH-adjusted q across the 5 metrics):", corr.round(4).to_string(index=False), ""]

    keep, dropped = _active_metrics(topo_df)
    if dropped:
        lines.append(f"Dropped zero-variance metrics (all topologies identical): {dropped}\n")

    model_raw, _ = multivariate_ols(topo_df, standardized=False)
    lines += ["Multivariate OLS accuracy ~ metrics (raw scale, HC3 robust SE):", str(model_raw.summary()), ""]

    model_std, _ = multivariate_ols(topo_df, standardized=True)
    lines += ["Multivariate OLS accuracy ~ metrics (standardized coefficients, HC3 robust SE):", str(model_std.summary()), ""]

    vif = vif_table(topo_df)
    lines += ["Variance Inflation Factor (VIF > 5 suggests notable multicollinearity):", vif.round(3).to_string(index=False), ""]

    survived = corr.loc[corr["pearson_q_bh"] < 0.05, "metric"].tolist()
    lines.append(f"Metrics surviving BH correction at q<0.05 (Pearson): {survived or 'none'}")

    (out_dir / "topology_level.txt").write_text("\n".join(lines))
    return survived


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — question-level panel
# ─────────────────────────────────────────────────────────────────────────────

def panel_logit(panel_df: pd.DataFrame, keep: list, use_fe: bool, cluster_col: str = "topology"):
    X = panel_df[keep].copy()
    if use_fe:
        fe = pd.get_dummies(panel_df["qi"], prefix="qi", drop_first=True, dtype=float)
        X = pd.concat([X, fe], axis=1)
    X = sm.add_constant(X.astype(float))
    model = sm.Logit(panel_df["match"], X).fit(
        cov_type="cluster", cov_kwds={"groups": panel_df[cluster_col]},
        disp=0, maxiter=200,
    )
    return model


def run_tier2(panel_df: pd.DataFrame, out_dir: Path) -> list:
    lines = [f"Tier 2 — question-level panel (n={len(panel_df)} = topologies x questions)", "=" * 70, ""]
    keep, dropped = _active_metrics(panel_df)
    if dropped:
        lines.append(f"Dropped zero-variance metrics: {dropped}\n")

    lines.append(f"match=1 rate: {panel_df['match'].mean():.4f}\n")

    survived = []
    try:
        model = panel_logit(panel_df, keep, use_fe=False)
        odds = np.exp(model.params).rename("odds_ratio")
        margeff = model.get_margeff()
        pvals = model.pvalues[keep]
        q = pd.Series(multipletests(pvals, method="fdr_bh")[1], index=keep)
        survived = [m for m in keep if q[m] < 0.05]

        lines += [
            "Logit match ~ metrics, SEs clustered by topology:",
            str(model.summary()), "",
            f"McFadden pseudo-R^2: {model.prsquared:.4f}", "",
            "Odds ratios:", odds.round(4).to_string(), "",
            "Marginal effects:", str(margeff.summary()), "",
            "BH-adjusted p-values (metric coefficients only):", q.round(4).to_string(), "",
            f"Metrics surviving BH correction at q<0.05: {survived or 'none'}", "",
        ]
    except Exception as exc:
        lines.append(f"Logit (no FE) failed: {exc!r}\n")

    try:
        model_fe = panel_logit(panel_df, keep, use_fe=True)
        lines += [
            "Logit match ~ metrics + C(qi) question fixed effects, SEs clustered by topology:",
            "(isolates the within-question effect of topology structure on correctness;",
            " question dummies not printed individually — only metric rows matter here)",
            "",
            str(model_fe.summary2().tables[1].loc[keep + ["const"]]) if hasattr(model_fe, "summary2") else str(model_fe.params[keep]),
            "",
            f"McFadden pseudo-R^2 (with FE): {model_fe.prsquared:.4f}", "",
        ]
    except Exception as exc:
        lines.append(f"Logit with question FE failed (common with many FE dummies / separation): {exc!r}\n")

    (out_dir / "question_level.txt").write_text("\n".join(lines))
    return survived


# ─────────────────────────────────────────────────────────────────────────────
# Plots (replicated from notebook/analysis.ipynb, redirected to the run's own folder)
# ─────────────────────────────────────────────────────────────────────────────

def save_plots(topo_df: pd.DataFrame, out_dir: Path) -> None:
    cmap = plt.get_cmap("tab20")
    colors = {t: cmap(i % 20) for i, t in enumerate(topo_df["topology"])}

    # scatter_plots.png
    fig, axes = plt.subplots(1, len(METRIC_COLS), figsize=(16, 3.5))
    for ax, metric in zip(axes, METRIC_COLS):
        x, y = topo_df[metric].values, topo_df["accuracy"].values
        for _, row in topo_df.iterrows():
            ax.scatter(row[metric], row["accuracy"], color=colors[row["topology"]], s=60, zorder=5)
        if len(x) > 2 and np.std(x) > 0:
            m, b = np.polyfit(x, y, 1)
            xline = np.linspace(x.min(), x.max(), 50)
            ax.plot(xline, m * xline + b, "--", color="#94a3b8", lw=1.2)
            r, p = stats.pearsonr(x, y)
            ax.set_title(f"{metric}\nr={r:.2f}, p={p:.2f}", fontsize=9)
        else:
            ax.set_title(metric, fontsize=9)
        ax.set_xlabel(metric, fontsize=9)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # mutual_info.png
    from sklearn.feature_selection import mutual_info_regression
    mi_rows = []
    for m in METRIC_COLS:
        x = topo_df[[m]].values
        mi = mutual_info_regression(x, topo_df["accuracy"].values, random_state=42)[0] if topo_df[m].std() > 0 else 0.0
        mi_rows.append({"metric": m, "mutual_info": mi})
    mi_df = pd.DataFrame(mi_rows).sort_values("mutual_info", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(mi_df["metric"], mi_df["mutual_info"], color="#10b981")
    ax.set_xlabel("Mutual Information (nats)")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_dir / "mutual_info.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # topk_accuracy.png
    def top_k_accuracy(df, metric, k):
        real_top = set(df.nlargest(k, "accuracy")["topology"])
        hi = len(real_top & set(df.nlargest(k, metric)["topology"]))
        lo = len(real_top & set(df.nsmallest(k, metric)["topology"]))
        return max(hi, lo) / k

    ks = [k for k in (1, 3, 5) if k <= len(topo_df)]
    topk_rows = [{"metric": m, **{f"top{k}": top_k_accuracy(topo_df, m, k) for k in ks}} for m in METRIC_COLS]
    topk_df = pd.DataFrame(topk_rows)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.2 * len(ks), 3.5), sharey=True)
    axes = [axes] if len(ks) == 1 else axes
    for ax, k in zip(axes, ks):
        vals = topk_df.set_index("metric")[f"top{k}"]
        ax.barh(vals.index, vals.values, color="#3B82F6")
        ax.set_xlim(0, 1.05)
        ax.axvline(1 / k, ls="--", color="#94a3b8", lw=1)
        ax.set_title(f"Top-{k} accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "topk_accuracy.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # radar.png
    radar_df = topo_df[METRIC_COLS].copy()
    radar_df = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-9)
    radar_df["topology"] = topo_df["topology"]
    N = len(METRIC_COLS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    for _, row in radar_df.iterrows():
        vals = row[METRIC_COLS].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[row["topology"]], lw=1.2, alpha=0.7)
    ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_COLS, fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "radar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run_id, results/<run_id> folder, or results/run_<run_id>.json")
    args = parser.parse_args()

    run_id, run_file, log_dir = _resolve(args.run)
    out_dir = ROOT / "results" / run_id / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    topo_df = load_topo_df(run_file)
    panel_df = load_panel_df(log_dir, topo_df)
    panel_df.to_csv(out_dir / "panel_data.csv", index=False)

    print(f"Topology-level: {len(topo_df)} topologies")
    print(f"Question-level panel: {len(panel_df)} rows\n")

    tier1_survived = run_tier1(topo_df, out_dir)
    tier2_survived = run_tier2(panel_df, out_dir)
    save_plots(topo_df, out_dir)

    vif = vif_table(topo_df)
    high_vif = vif.loc[vif["VIF"] > 5, "metric"].tolist()

    print("=== Summary ===")
    print(f"Tier 1 (topology-level) survives BH q<0.05: {tier1_survived or 'none'}")
    print(f"Tier 2 (question-level panel) survives BH q<0.05: {tier2_survived or 'none'}")
    if high_vif:
        print(f"VIF > 5 (multicollinearity warning): {high_vif}")
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
