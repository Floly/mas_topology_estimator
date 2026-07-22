"""
CLI version of notebook/openrouter_run.ipynb — runs the topology PoC through
OpenRouter (or any OpenAI-compatible endpoint) using a YAML config.

Usage:
    python scripts/run_openrouter.py --config configs/openrouter_run.yaml
    python scripts/run_openrouter.py --config configs/openrouter_run.yaml --stub --n-questions 2
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
from dotenv import load_dotenv

from mas.agent import Agent, AgentConfig
from mas.prompts import SYSTEM_PROMPTS, parse_answer_str
from mas.runner import MASRunner
from metrics.graph_metrics import TopologyMetrics
from topologies.definitions import HYBRID_ROLES, get_all_topologies, get_topologies

_KNOWN_ROLES = set(SYSTEM_PROMPTS.keys())
_CYCLE = ["solver", "critic", "aggregator"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading + answer matching (ported from notebook/openrouter_run.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(dataset: str, n: int) -> list:
    if dataset == "gsm8k":
        from datasets import load_dataset
        try:
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
        except Exception:
            import urllib.request
            url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
            with urllib.request.urlopen(url) as r:
                ds = [json.loads(l) for l in r.read().decode().splitlines()[:n]]
        return [
            {
                "question": item["question"],
                "answer": re.search(r"####\s*(-?\d[\d,]*)", item["answer"]).group(1).replace(",", ""),
            }
            for item in ds
        ]
    if dataset == "math500":
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split=f"test[:{n}]")
        return [
            {"question": item["problem"], "answer": item["answer"], "level": item.get("level", "")}
            for item in ds
        ]
    if dataset == "hle":
        from datasets import load_dataset
        ds = load_dataset("cais/hle", split=f"test[:{n}]")
        return [
            {"question": item["question"], "answer": item["answer"], "level": item.get("level", "")}
            for item in ds
        ]
    if dataset == "gpqa":
        from datasets import load_dataset
        ds = load_dataset("fingertap/GPQA-Diamond", split=f"test[:{n}]")
        return [
            {"question": item["question"], "answer": item["answer"], "level": item.get("level", "")}
            for item in ds
        ]
    if dataset == "gaia":
        from datasets import load_dataset
        ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=f"validation[:{n}]")
        return [
            {"question": item["Question"], "answer": item["Final answer"], "level": item.get("level", "")}
            for item in ds
        ]
    raise ValueError(f"Unknown dataset: {dataset!r}")


def _normalize(s: str) -> str:
    s = re.sub(r"\\left\s*[\(\[\{]", "", s.strip())
    s = re.sub(r"\\right\s*[\)\]\}]", "", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    return re.sub(r"[$\\\s]+", " ", s).strip().lower()


def _to_float(s: str):
    try:
        return float(s.split("/")[0]) / float(s.split("/")[1]) if s.count("/") == 1 else float(s)
    except Exception:
        return None


def answers_match(pred, gt) -> bool:
    if pred is None:
        return False
    p, g = _normalize(pred), _normalize(gt)
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    return pf is not None and gf is not None and abs(pf - gf) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Agent building (ported from notebook/openrouter_run.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_role(node: str, idx: int) -> str:
    if node == "agg":
        return "aggregator"
    if node in _KNOWN_ROLES:
        return node
    for r in _KNOWN_ROLES:
        if node.startswith(r):
            return r
    return _CYCLE[idx % len(_CYCLE)]


def build_agents(graph, topo_name: str, cfg: dict) -> dict:
    order = [n for n in nx.topological_sort(graph) if n != "task"]
    hybrid_map = HYBRID_ROLES.get(topo_name, {})
    return {
        node: Agent(AgentConfig(
            agent_id=node,
            role=hybrid_map.get(node) or _infer_role(node, order.index(node)),
            model=cfg["model"],
            stub=cfg["stub"],
            base_url=None if cfg["stub"] else cfg["base_url"],
            api_key_env=cfg["api_key_env"],
            temperature=cfg["temperature"],
        ))
        for node in order
    }


# ─────────────────────────────────────────────────────────────────────────────
# Config loading + CLI overrides
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path, args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if args.stub:
        cfg["stub"] = True
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.n_questions is not None:
        cfg["n_questions"] = args.n_questions
    if args.all_topologies is not None:
        cfg["all_topologies"] = args.all_topologies
    if args.model is not None:
        cfg["model"] = args.model
    cfg.setdefault("temperature", None)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/openrouter_run.yaml")
    parser.add_argument("--stub", action="store_true", help="force stub mode regardless of config")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--all-topologies", dest="all_topologies", action="store_true", default=None)
    parser.add_argument("--no-all-topologies", dest="all_topologies", action="store_false")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), args)

    print(f"ROOT       : {ROOT}")
    print(f"model      : {cfg['model']}")
    print(f"base_url   : {cfg['base_url']}")
    print(f"dataset    : {cfg['dataset']}  n={cfg['n_questions']}")
    print(f"stub       : {cfg['stub']}")

    # ── env / API key ────────────────────────────────────────────────────
    for p in [ROOT, ROOT.parent, Path(".").resolve()]:
        if (p / ".env").exists():
            load_dotenv(p / ".env", override=True)
            print(f".env loaded from {p / '.env'}")
            break
    else:
        print("WARNING: .env not found — set the API key env var manually")

    api_key = os.environ.get(cfg["api_key_env"])
    if api_key:
        print(f"{cfg['api_key_env']} set: {api_key[:8]}...")
    elif not cfg["stub"]:
        raise EnvironmentError(
            f"{cfg['api_key_env']} not set. Add it to .env or export it."
        )
    else:
        print(f"{cfg['api_key_env']} not set (OK in stub mode)")

    # ── data ─────────────────────────────────────────────────────────────
    questions = load_questions(cfg["dataset"], cfg["n_questions"])
    print(f"Loaded     : {len(questions)} questions")
    if questions:
        print(f"Example    : {questions[0]['question'][:100]}...")
        print(f"GT         : {questions[0]['answer']}")

    # ── topologies ───────────────────────────────────────────────────────
    topologies = get_all_topologies() if cfg["all_topologies"] else get_topologies(n_agents=3)
    metrics_engine = TopologyMetrics()
    print(f"Topologies : {len(topologies)}")

    # ── sweep ────────────────────────────────────────────────────────────
    results = []
    failed_topologies = []

    for topo_name, graph in topologies.items():
        t0 = time.perf_counter()
        agents = build_agents(graph, topo_name, cfg)
        runner = MASRunner(graph, agents)
        metrics = metrics_engine.compute(graph, topo_name)

        correct = 0
        run_tokens = 0
        n_errors = 0

        for item in questions:
            try:
                out, q_tokens = runner.run(item["question"])
            except Exception as exc:
                n_errors += 1
                print(f"  [{topo_name}] question failed after retries, counting as wrong: {exc}")
                continue
            run_tokens += q_tokens
            pred = parse_answer_str(out)
            if answers_match(pred, item["answer"]):
                correct += 1
        

        acc = correct / len(questions) if questions else 0.0
        dur = round(time.perf_counter() - t0, 2)
        acc_per1k = round(acc / (run_tokens / 1000), 6) if run_tokens > 0 else None

        print(f"{topo_name:30s}  acc={acc:.2f}  tokens={run_tokens}  errors={n_errors}  {dur}s")
        if n_errors:
            failed_topologies.append((topo_name, n_errors))

        results.append({
            "topology": topo_name,
            "accuracy": acc,
            "total_tokens": run_tokens,
            "accuracy_per_1k_tokens": acc_per1k,
            "duration_sec": dur,
            "n_errors": n_errors,
            "metrics": {
                "diameter": metrics.diameter,
                "avg_degree": metrics.avg_degree,
                "structural_entropy": metrics.structural_entropy,
                "spectral_gap": metrics.spectral_gap,
                "task_centrality": metrics.task_centrality,
            },
        })

    print("\nDone.")
    if failed_topologies:
        print(f"Topologies with errors (post-retry): {failed_topologies}")

    # ── save ─────────────────────────────────────────────────────────────
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    run_ts = datetime.now(timezone.utc)
    run_id = run_ts.strftime("%Y%m%d_%H%M%S")

    run_doc = {
        "meta": {
            "run_id": run_id,
            "timestamp": run_ts.isoformat(),
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "dataset": cfg["dataset"],
            "n_questions": len(questions),
            "stub": cfg["stub"],
            "n_topologies": len(topologies),
            "mean_accuracy": round(sum(r["accuracy"] for r in results) / len(results), 4) if results else 0.0,
            "total_tokens": sum(r["total_tokens"] for r in results),
        },
        "results": results,
    }

    run_file = out_dir / f"run_{run_id}.json"
    run_file.write_text(json.dumps(run_doc, indent=2, ensure_ascii=False))
    (out_dir / "results.json").write_text(json.dumps(run_doc, indent=2, ensure_ascii=False))

    index_path = out_dir / "runs_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index.append({**run_doc["meta"], "results_file": str(run_file)})
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(f"Run ID   : {run_id}")
    print(f"Saved    : {run_file}")
    print(json.dumps(run_doc["meta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
