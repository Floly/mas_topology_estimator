"""
Runs the full PoC:
  1. Loads questions (GSM8K or MATH-500)
  2. For each topology runs MAS
  3. Computes graph metrics
  4. Saves results/run_<timestamp>.json  +  updates results/runs_index.json

Usage:
    python scripts/run_poc.py --model gpt-3.5-turbo --n-questions 20
    python scripts/run_poc.py --dataset math500 --n-questions 10
    python scripts/run_poc.py --stub --n-questions 5
    python scripts/run_poc.py --all-topologies --n-questions 20
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
import networkx as nx

from topologies.definitions import get_topologies, get_all_topologies, HYBRID_ROLES
from mas.agent import Agent, AgentConfig
from mas.runner import MASRunner
from mas.prompts import parse_answer, parse_answer_str, SYSTEM_PROMPTS
from metrics.graph_metrics import TopologyMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_level(level) -> int:
    """Normalize MATH-500 level field (int, str digit, or 'Level N') to int."""
    if isinstance(level, int):
        return level
    s = str(level).strip()
    if s.isdigit():
        return int(s)
    m = re.search(r"\d+", s)
    return int(m.group()) if m else -1


def load_questions(dataset: str, n: int, levels: set | None = None) -> list:
    """
    Returns list of {"question": str, "answer": str}.
    answer is always a plain string (integer text for GSM8K,
    possibly LaTeX expression for MATH-500).
    levels: set of ints for math500 level filter (None = no filter).
    """
    if dataset == "gsm8k":
        try:
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
        except Exception:
            import urllib.request, json as _j
            url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
            print("  HuggingFace unavailable, downloading from GitHub...")
            with urllib.request.urlopen(url) as r:
                lines = r.read().decode().splitlines()
            ds = [_j.loads(l) for l in lines[:n]]
        return [
            {
                "question": item["question"],
                "answer":   re.search(r"####\s*(-?\d[\d,]*)", item["answer"]).group(1).replace(",", ""),
            }
            for item in ds
        ]

    if dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = list(ds)
        if levels:
            items = [item for item in items if _parse_level(item.get("level")) in levels]
        items = items[:n]
        return [
            {
                "question": item["problem"],
                "answer":   item["answer"],
                "subject":  item.get("subject", ""),
                "level":    item.get("level", ""),
            }
            for item in items
        ]

    if dataset == "bbh_logic":
        items = []
        try:
            ds5 = load_dataset("lukaemon/bbh", "logical_deduction_five_objects", split="test")
            items = list(ds5)
        except Exception:
            pass
        if len(items) < n:
            try:
                ds3 = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
                items += list(ds3)[: n - len(items)]
            except Exception:
                pass
        return [
            {"question": item["input"], "answer": item["target"]}
            for item in items[:n]
        ]

    raise ValueError(f"Unknown dataset: {dataset!r}. Choose 'gsm8k', 'math500', or 'bbh_logic'.")


# ─────────────────────────────────────────────────────────────────────────────
# Answer comparison
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip LaTeX wrappers and whitespace for comparison."""
    s = s.strip()
    # remove \left( ... \right) wrappers
    s = re.sub(r"\\left\s*[\(\[\{]", "", s)
    s = re.sub(r"\\right\s*[\)\]\}]", "", s)
    # normalize \frac{a}{b} → a/b
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # remove $, \, and extra spaces
    s = re.sub(r"[$\\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def _to_float(s: str):
    try:
        # handle simple fractions like 1/2
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def _extract_letter(s: str):
    """Return lowercased letter from '(C)' patterns, or None."""
    m = re.search(r"\(([A-Za-z])\)", s)
    return m.group(1).lower() if m else None


def answers_match(pred_raw: str, gt: str) -> bool:
    """Compare model answer string to ground-truth string."""
    if pred_raw is None:
        return False
    # letter-option matching for BBH-style answers like (C)
    pl = _extract_letter(pred_raw)
    gl = _extract_letter(gt)
    if pl is not None and gl is not None:
        return pl == gl
    # numeric / LaTeX path (unchanged)
    p = _normalize(pred_raw)
    g = _normalize(gt)
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-6
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Agent building
# ─────────────────────────────────────────────────────────────────────────────

_KNOWN_ROLES = set(SYSTEM_PROMPTS.keys())
_ROLES_CYCLE = ["solver", "critic", "aggregator"]


def _infer_role(node: str, index: int) -> str:
    if node == "agg":
        return "aggregator"
    if node in _KNOWN_ROLES:
        return node
    for role in _KNOWN_ROLES:
        if node.startswith(role):
            return role
    return _ROLES_CYCLE[index % len(_ROLES_CYCLE)]


def build_agents(
    graph: nx.DiGraph,
    topo_name: str,
    model: str,
    stub: bool,
    base_url: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float | None = None,
) -> dict:
    order      = [n for n in nx.topological_sort(graph) if n != "task"]
    hybrid_map = HYBRID_ROLES.get(topo_name, {})
    return {
        node: Agent(AgentConfig(
            agent_id=node,
            role=hybrid_map.get(node) or _infer_role(node, order.index(node)),
            model=model,
            stub=stub,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
        ))
        for node in order
    }


# ─────────────────────────────────────────────────────────────────────────────
# Index helper
# ─────────────────────────────────────────────────────────────────────────────

def _update_index(index_path: Path, entry: dict) -> None:
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index.append(entry)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for p in [Path(__file__).parent.parent, Path(__file__).parent]:
        if (p / ".env").exists():
            load_dotenv(p / ".env", override=True)
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          default="gpt-3.5-turbo")
    parser.add_argument("--n-questions",    type=int, default=20)
    parser.add_argument("--dataset",        default="gsm8k",
                        choices=["gsm8k", "math500", "bbh_logic"],
                        help="Dataset to evaluate on")
    parser.add_argument("--levels",         default="4,5",
                        help="Comma-separated MATH-500 difficulty levels to keep (default: 4,5)")
    parser.add_argument("--stub",           action="store_true")
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--base-url",       default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env",    default="OPENAI_API_KEY",
                        help="Env var name holding the API key")
    parser.add_argument("--temperature",    type=float, default=None,
                        help="Override model temperature")
    args = parser.parse_args()
    levels = {int(l) for l in args.levels.split(",") if l.strip()} if args.dataset == "math500" else None

    run_ts  = datetime.now(timezone.utc)
    run_id  = run_ts.strftime("%Y%m%d_%H%M%S")

    print(f"Run ID  : {run_id}")
    print(f"Dataset : {args.dataset}  ({args.n_questions} questions)")
    questions = load_questions(args.dataset, args.n_questions, levels=levels)
    print(f"Loaded  : {len(questions)} questions")
    print(f"Example : {questions[0]['question'][:80]}...")
    print(f"GT      : {questions[0]['answer']}")

    topologies     = get_all_topologies() if args.all_topologies else get_topologies(n_agents=3)
    metrics_engine = TopologyMetrics()

    print(f"\nTopologies: {len(topologies)} | stub={args.stub} | model={args.model}\n")

    results = []
    run_t0  = time.perf_counter()

    for topo_name, graph in topologies.items():
        topo_t0 = time.perf_counter()
        agents  = build_agents(
            graph, topo_name, args.model, args.stub,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            temperature=args.temperature,
        )
        runner  = MASRunner(graph, agents)
        metrics = metrics_engine.compute(graph, topo_name)

        correct = 0
        run_tokens = 0
        for i, item in enumerate(questions):
            out, q_tokens = runner.run(item["question"])
            run_tokens += q_tokens
            pred = parse_answer_str(out)
            if answers_match(pred, item["answer"]):
                correct += 1
            if args.stub and i == 0:
                print(f"  [{topo_name}] pred={pred!r}  gt={item['answer']!r}")

        topo_sec = round(time.perf_counter() - topo_t0, 2)
        accuracy = correct / len(questions)
        acc_per_1k = round(accuracy / (run_tokens / 1000), 6) if run_tokens > 0 else None
        print(f"{topo_name:30s}  acc={accuracy:.2f}  ({correct}/{len(questions)})  tokens={run_tokens}  {topo_sec}s")

        results.append({
            "topology":              topo_name,
            "accuracy":              accuracy,
            "total_tokens":          run_tokens,
            "accuracy_per_1k_tokens": acc_per_1k,
            "duration_sec":          topo_sec,
            "metrics": {
                "diameter":           metrics.diameter,
                "avg_degree":         metrics.avg_degree,
                "structural_entropy": metrics.structural_entropy,
                "spectral_gap":       metrics.spectral_gap,
                "task_centrality":    metrics.task_centrality,
                "flow_entropy":       metrics.flow_entropy,
                "von_neumann_entropy": metrics.von_neumann_entropy,
            },
        })

    total_sec = round(time.perf_counter() - run_t0, 2)
    mean_acc  = round(sum(r["accuracy"] for r in results) / len(results), 4)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    run_file = results_dir / f"run_{run_id}.json"

    run_doc = {
        "meta": {
            "run_id":         run_id,
            "timestamp":      run_ts.isoformat(),
            "model":          args.model,
            "base_url":       args.base_url,
            "api_key_env":    args.api_key_env,
            "dataset":        args.dataset,
            "levels":         args.levels if args.dataset == "math500" else None,
            "n_questions":    len(questions),
            "stub":           args.stub,
            "all_topologies": args.all_topologies,
            "n_topologies":   len(topologies),
            "duration_sec":   total_sec,
            "mean_accuracy":  mean_acc,
        },
        "results": results,
    }

    run_file.write_text(json.dumps(run_doc, indent=2, ensure_ascii=False))
    (results_dir / "results.json").write_text(json.dumps(run_doc, indent=2, ensure_ascii=False))
    _update_index(results_dir / "runs_index.json", {**run_doc["meta"], "results_file": str(run_file)})

    print(f"\nTotal: {total_sec}s  |  mean accuracy: {mean_acc:.4f}")
    print(f"Saved → {run_file}")
    print(f"Index → {results_dir / 'runs_index.json'}")


if __name__ == "__main__":
    main()
