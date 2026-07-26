"""
CLI version of notebook/openrouter_run.ipynb — runs the topology PoC through
OpenRouter (or any OpenAI-compatible endpoint) using a YAML config.

Usage:
    python scripts/run_openrouter.py --config configs/openrouter_run.yaml
    python scripts/run_openrouter.py --config configs/openrouter_run.yaml --stub --n-questions 2
    python scripts/run_openrouter.py --config configs/openrouter_run.yaml --resume results/run_20260722_120000.json
"""
import argparse
import asyncio
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
# Incremental save / resume
# ─────────────────────────────────────────────────────────────────────────────

def _write_run(run_doc: dict, run_file: Path, out_dir: Path) -> None:
    """Write run_doc to run_file and mirror it to results/results.json (latest run)."""
    text = json.dumps(run_doc, indent=2, ensure_ascii=False)
    run_file.write_text(text)
    (out_dir / "results.json").write_text(text)


def _refresh_meta(run_doc: dict, results: list, status: str) -> None:
    m = run_doc["meta"]
    m["status"] = status
    m["n_completed"] = len(results)
    m["mean_accuracy"] = round(sum(r["accuracy"] for r in results) / len(results), 4) if results else 0.0
    m["total_tokens"] = sum(r["total_tokens"] for r in results)


def _write_topo_log(log_dir: Path, topo_name: str, n_questions: int, records: dict) -> None:
    """Per-topology per-question detail log — results/<run_id>/<topo_name>.json."""
    doc = {
        "topology": topo_name,
        "n_questions": n_questions,
        "records": [records[qi] for qi in sorted(records)],
    }
    (log_dir / f"{topo_name}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))


def _load_topo_log(log_dir: Path, topo_name: str) -> dict:
    """
    Returns {qi: record} for previously-successful questions, for resume.
    A record only counts as done if it has no error AND a non-null parsed
    answer — a call can succeed (error is None) yet still fail to produce a
    parseable "ANSWER: <number>", which must be retried, not treated as done.
    """
    path = log_dir / f"{topo_name}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    return {
        r["qi"]: r for r in doc.get("records", [])
        if r.get("error") is None and r.get("predicted") is not None
    }


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
            debug=cfg["debug"],
        ))
        for node in order
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flat concurrent sweep — every (topology, question) pair shares one pool
# ─────────────────────────────────────────────────────────────────────────────

async def _run_sweep(
    topologies: dict,
    questions: list,
    cfg: dict,
    metrics_engine: TopologyMetrics,
    results: list,
    completed: set,
    run_doc: dict,
    run_file: Path,
    out_dir: Path,
    log_dir: Path,
) -> list:
    """
    Flattens every (topology, question) pair not already completed into one
    global pool bounded by a single cfg['max_concurrency'] semaphore, so a
    topology with only a handful of questions left never starves the pool
    while other topologies still have untouched work. Each question gets its
    own fresh agents/runner (Agent.total_tokens is mutated in place per call,
    so sharing Agent instances across concurrently-running questions would
    race on that counter). Per-question results are written to
    results/<run_id>/<topo_name>.json as each one finishes; once a
    topology's last pending question lands, its aggregate row is appended to
    `results` and the run file is saved incrementally — same as before, just
    driven by "last pending question done" instead of "loop iteration done".
    Returns the list of (topo_name, n_errors) for topologies that had errors.
    """
    sem = asyncio.Semaphore(cfg["max_concurrency"])
    global_state = {"active": 0, "queued": 0}
    topo_state: dict = {}
    failed_topologies: list = []

    def finalize(topo_name: str) -> None:
        state = topo_state[topo_name]
        records = state["records"]
        n = len(questions)
        correct = sum(1 for r in records.values() if r["match"])
        run_tokens = sum(r["tokens"] for r in records.values())
        n_errors = sum(1 for r in records.values() if r["error"] is not None)
        acc = correct / n if n else 0.0
        dur = round(state["t_end"] - state["t_start"], 2) if state["t_start"] is not None else 0.0
        acc_per1k = round(acc / (run_tokens / 1000), 6) if run_tokens > 0 else None
        metrics = metrics_engine.compute(state["graph"], topo_name)

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
        completed.add(topo_name)
        _refresh_meta(run_doc, results, status="in_progress")
        _write_run(run_doc, run_file, out_dir)

    async def run_one(topo_name: str, qi: int, item: dict) -> None:
        state = topo_state[topo_name]
        graph = state["graph"]
        if state["t_start"] is None:
            state["t_start"] = time.perf_counter()

        global_state["queued"] += 1
        wait_t0 = time.perf_counter()
        await sem.acquire()
        wait_dur = time.perf_counter() - wait_t0
        global_state["queued"] -= 1
        global_state["active"] += 1
        print(f"  [conc:{topo_name}] q{qi} start  active={global_state['active']}/{cfg['max_concurrency']} queued={global_state['queued']}")

        try:
            agents = build_agents(graph, topo_name, cfg)
            runner = MASRunner(graph, agents)
            q_t0 = time.perf_counter()
            out, q_tokens = await asyncio.to_thread(runner.run, item["question"])
            q_dur = time.perf_counter() - q_t0
            pred = parse_answer_str(out)
            is_match = answers_match(pred, item["answer"])
            record = {
                "qi": qi, "question": item["question"], "ground_truth": item["answer"],
                "predicted": pred, "match": is_match, "tokens": q_tokens,
                "duration_sec": round(q_dur, 3), "wait_sec": round(wait_dur, 3), "error": None,
            }
            if cfg["debug"]:
                q_text = item["question"].replace("\n", " ")
                q_text = q_text[:150] + "..." if len(q_text) > 150 else q_text
                print(
                    f"  [debug:{topo_name}] q{qi}: {q_text}\n"
                    f"  [debug:{topo_name}] q{qi}: pred={pred!r} gt={item['answer']!r} "
                    f"match={is_match} tokens={q_tokens} wait={wait_dur:.2f}s call={q_dur:.2f}s"
                )
        except Exception as exc:
            print(f"  [{topo_name}] question {qi} failed after retries, counting as wrong: {exc}")
            record = {
                "qi": qi, "question": item["question"], "ground_truth": item["answer"],
                "predicted": None, "match": False, "tokens": 0,
                "duration_sec": None, "wait_sec": round(wait_dur, 3), "error": str(exc),
            }
        finally:
            global_state["active"] -= 1
            sem.release()
            print(f"  [conc:{topo_name}] q{qi} done   active={global_state['active']}/{cfg['max_concurrency']} queued={global_state['queued']}")

        state["records"][qi] = record
        state["pending"].discard(qi)
        state["t_end"] = time.perf_counter()
        _write_topo_log(log_dir, topo_name, len(questions), state["records"])
        if not state["pending"]:
            finalize(topo_name)

    tasks = []
    for topo_name, graph in topologies.items():
        if topo_name in completed:
            continue
        prior = _load_topo_log(log_dir, topo_name)
        pending = {qi for qi in range(len(questions)) if qi not in prior}
        topo_state[topo_name] = {
            "graph": graph, "records": dict(prior), "pending": pending,
            "t_start": None, "t_end": None,
        }
        if not pending:
            finalize(topo_name)
            continue
        for qi in sorted(pending):
            tasks.append(run_one(topo_name, qi, questions[qi]))

    if tasks:
        await asyncio.gather(*tasks)

    return failed_topologies


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
    if args.debug:
        cfg["debug"] = True
    if args.max_concurrency is not None:
        cfg["max_concurrency"] = args.max_concurrency
    cfg.setdefault("temperature", None)
    cfg.setdefault("debug", False)
    cfg.setdefault("max_concurrency", 8)
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
    parser.add_argument("--debug", action="store_true", help="force verbose per-question/per-agent logging")
    parser.add_argument("--resume", default=None, help="path to an existing run_*.json; skips topologies already present in it")
    parser.add_argument("--max-concurrency", type=int, default=None, help="max concurrent in-flight LLM calls per topology")
    args = parser.parse_args()

    cfg = load_config(Path(args.config), args)

    resume_run_doc = None
    if args.resume:
        resume_run_doc = json.loads(Path(args.resume).read_text())
        meta = resume_run_doc["meta"]
        # The run's own meta is the experiment description — it must win over
        # whatever the current yaml/CLI say, or a drifted config (e.g. a
        # different n_questions) would misalign question indices against the
        # existing per-topology logs.
        basic_n = len(get_topologies(n_agents=3))
        overrides = {
            "dataset": meta["dataset"],
            "n_questions": meta["n_questions"],
            "model": meta["model"],
            "base_url": meta["base_url"],
            "stub": meta["stub"],
            "all_topologies": meta["n_topologies"] != basic_n,
        }
        for key, value in overrides.items():
            if cfg.get(key) != value:
                print(f"  [resume] {key}: {cfg.get(key)!r} -> {value!r} (from experiment description)")
            cfg[key] = value

    print(f"ROOT       : {ROOT}")
    print(f"model      : {cfg['model']}")
    print(f"base_url   : {cfg['base_url']}")
    print(f"dataset    : {cfg['dataset']}  n={cfg['n_questions']}")
    print(f"stub       : {cfg['stub']}")
    print(f"debug      : {cfg['debug']}")
    print(f"concurrency: {cfg['max_concurrency']}")

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

    # ── resume / init run doc ───────────────────────────────────────────
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    if args.resume:
        run_file = Path(args.resume)
        run_doc = resume_run_doc
        run_id = run_doc["meta"]["run_id"]
        results = run_doc["results"]
        completed = {r["topology"] for r in results}

        log_dir = out_dir / run_id
        reopened = []
        for topo_name in list(completed):
            prior = _load_topo_log(log_dir, topo_name)
            if len(prior) < len(questions):
                completed.discard(topo_name)
                results[:] = [r for r in results if r["topology"] != topo_name]
                reopened.append(topo_name)
        if reopened:
            print(f"  [resume] reopening {len(reopened)} topologies with bad/missing records: {reopened}")

        print(f"Resuming   : {run_id} ({len(completed)}/{len(topologies)} topologies fully done)")
    else:
        run_ts = datetime.now(timezone.utc)
        run_id = run_ts.strftime("%Y%m%d_%H%M%S")
        run_file = out_dir / f"run_{run_id}.json"
        results = []
        completed = set()
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
                "status": "in_progress",
            },
            "results": results,
        }

    log_dir = out_dir / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── sweep ────────────────────────────────────────────────────────────
    failed_topologies = asyncio.run(
        _run_sweep(topologies, questions, cfg, metrics_engine, results, completed, run_doc, run_file, out_dir, log_dir)
    )

    print("\nDone.")
    if failed_topologies:
        print(f"Topologies with errors (post-retry): {failed_topologies}")

    # ── finalize ─────────────────────────────────────────────────────────
    _refresh_meta(run_doc, results, status="done")
    _write_run(run_doc, run_file, out_dir)

    index_path = out_dir / "runs_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    if not any(e.get("run_id") == run_id for e in index):
        index.append({**run_doc["meta"], "results_file": str(run_file)})
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))

    print(f"Run ID   : {run_id}")
    print(f"Saved    : {run_file}")
    print(json.dumps(run_doc["meta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
