"""
Post-hoc re-grading of a finished run: many "wrong" answers are actually
correct but failed the live string-match because of LaTeX/formatting
differences (\\text{Evelyn} vs Evelyn, "1, -2" vs "1,-2", \\sqrt{51} vs
sqrt(51) vs √51, 90^\\circ vs 90, ...). This script re-checks every
match=False record in a run's per-topology logs with a more thorough,
regex-based normalizer (plus a couple of narrow extra tiers), and writes a
NEW run + log folder with "_fix" appended to the run id — it never modifies
the source run.

Usage:
    python scripts/error_fixer.py 20260725_101948
    python scripts/error_fixer.py results/20260725_101948
    python scripts/error_fixer.py results/run_20260725_101948.json
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────────────────────────────────────
# Normalization tiers
# ─────────────────────────────────────────────────────────────────────────────

def _canonicalize(s: str) -> str:
    """Strip LaTeX/formatting noise down to a bare comparable token."""
    s = s.strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)                # \text{Evelyn} -> Evelyn
    s = re.sub(r"\\left\s*", "", s)                             # \left( -> (
    s = re.sub(r"\\right\s*", "", s)                            # \right) -> )
    s = re.sub(r"\\[()\[\]]", "", s)                            # \( \) \[ \] delimiters
    s = s.replace("$", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)  # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)           # \sqrt{51} -> sqrt(51)
    s = re.sub(r"\\sqrt\s*([0-9]+)", r"sqrt(\1)", s)            # \sqrt51 (no braces)
    s = re.sub(r"√\s*([0-9]+)", r"sqrt(\1)", s)                 # √51 -> sqrt(51)
    s = s.replace("\\pi", "pi").replace("π", "pi")
    s = re.sub(r"(?<=\d)\s*\^?\\circ", "", s)                   # 90^\circ / 90\circ -> 90
    s = re.sub(r"(?<=\d)\s*°", "", s)                           # 90° -> 90
    s = re.sub(r"^[a-zA-Z]\w*\s*=\s*", "", s)                   # "x=5" -> "5"
    s = s.replace("\\", "").replace("{", "").replace("}", "")   # leftover LaTeX noise
    s = re.sub(r"\s+", "", s)                                   # formatting-only spacing never matters
    return s.lower()


_SAFE_NAMES = {"pi": math.pi, "sqrt": math.sqrt}
_NUMERIC_CHARSET = re.compile(r"[0-9+\-*/(). a-z]+")


def _safe_numeric_eval(expr: str):
    if not expr or not _NUMERIC_CHARSET.fullmatch(expr):
        return None
    try:
        return eval(expr, {"__builtins__": {}}, _SAFE_NAMES)
    except Exception:
        return None


def tiered_match(pred: str, gt: str) -> str | None:
    """Returns the tier name that matched, or None if still a genuine mismatch."""
    p, g = _canonicalize(pred), _canonicalize(gt)
    if not p or not g:
        return None
    if p == g:
        return "exact_normalized"

    p_tokens = sorted(t for t in p.split(",") if t)
    g_tokens = sorted(t for t in g.split(",") if t)
    if len(p_tokens) > 1 and p_tokens == g_tokens:
        return "multiset"

    pv, gv = _safe_numeric_eval(p), _safe_numeric_eval(g)
    if pv is not None and gv is not None:
        if abs(pv - gv) <= max(1e-6, abs(gv) * 1e-3):
            return "numeric_eval"

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Run resolution + fixing
# ─────────────────────────────────────────────────────────────────────────────

def _resolve(arg: str) -> tuple[str, Path, Path]:
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


def fix_topology(records: list) -> tuple[list, dict]:
    """Re-grades one topology's records. Returns (new_records, tier_counts)."""
    tier_counts: dict = {}
    new_records = []
    for r in records:
        r = dict(r)
        if not r["match"] and r.get("predicted") is not None:
            tier = tiered_match(r["predicted"], r["ground_truth"])
            if tier:
                r["original_match"] = False
                r["match"] = True
                r["fix_tier"] = tier
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
        new_records.append(r)
    return new_records, tier_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="run_id, results/<run_id> folder, or results/run_<run_id>.json")
    args = parser.parse_args()

    run_id, run_file, log_dir = _resolve(args.run)
    run_doc = json.loads(run_file.read_text())
    orig_rows = {r["topology"]: r for r in run_doc["results"]}

    fix_run_id = f"{run_id}_fix"
    out_dir = ROOT / "results"
    fix_log_dir = out_dir / fix_run_id
    fix_log_dir.mkdir(parents=True, exist_ok=True)

    fixed_results = []
    total_tier_counts: dict = {}
    print(f"Source     : {run_file}  ({log_dir})")
    print(f"Output     : {out_dir / f'run_{fix_run_id}.json'}  ({fix_log_dir})\n")

    for log_path in sorted(log_dir.glob("*.json")):
        topo_name = log_path.stem
        doc = json.loads(log_path.read_text())
        new_records, tier_counts = fix_topology(doc["records"])

        (fix_log_dir / f"{topo_name}.json").write_text(json.dumps(
            {"topology": topo_name, "n_questions": doc["n_questions"], "records": new_records},
            indent=2, ensure_ascii=False,
        ))

        n_reclassified = sum(tier_counts.values())
        for tier, n in tier_counts.items():
            total_tier_counts[tier] = total_tier_counts.get(tier, 0) + n

        orig_row = orig_rows.get(topo_name)
        n = len(new_records)
        correct = sum(1 for r in new_records if r["match"])
        acc = correct / n if n else 0.0

        if orig_row is not None:
            row = dict(orig_row)
            row["accuracy"] = acc
            row["n_errors"] = sum(1 for r in new_records if r.get("error") is not None)
            row["accuracy_per_1k_tokens"] = (
                round(acc / (row["total_tokens"] / 1000), 6) if row["total_tokens"] > 0 else None
            )
            fixed_results.append(row)
            old_acc = orig_row["accuracy"]
        else:
            print(f"  WARNING: {topo_name} has no row in the source aggregate — excluded from fixed results.json")
            old_acc = None

        breakdown = ", ".join(f"{t}={c}" for t, c in tier_counts.items()) or "none"
        old_acc_s = f"{old_acc:.2f}" if old_acc is not None else "n/a"
        print(f"{topo_name:30s}  acc {old_acc_s} -> {acc:.2f}  reclassified={n_reclassified:3d}  ({breakdown})")

    fix_doc = {
        "meta": {
            **run_doc["meta"],
            "run_id": fix_run_id,
            "fixed_from": run_id,
            "mean_accuracy": round(sum(r["accuracy"] for r in fixed_results) / len(fixed_results), 4) if fixed_results else 0.0,
            "n_completed": len(fixed_results),
            "status": "done",
        },
        "results": fixed_results,
    }
    (out_dir / f"run_{fix_run_id}.json").write_text(json.dumps(fix_doc, indent=2, ensure_ascii=False))

    total = sum(total_tier_counts.values())
    breakdown = ", ".join(f"{t}={c}" for t, c in total_tier_counts.items()) or "none"
    print(f"\nTotal reclassified: {total}  ({breakdown})")
    print(f"mean_accuracy: {run_doc['meta']['mean_accuracy']} -> {fix_doc['meta']['mean_accuracy']}")
    print(f"Saved: {out_dir / f'run_{fix_run_id}.json'}")


if __name__ == "__main__":
    main()
