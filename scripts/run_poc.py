"""
Runs the full PoC:
  1. Loads GSM8K questions
  2. For each topology runs MAS
  3. Computes graph metrics
  4. Saves results/results.json

Usage:
    python scripts/run_poc.py --model gpt-3.5-turbo --n-questions 20
    python scripts/run_poc.py --stub --n-questions 5        # no API calls
    python scripts/run_poc.py --all-topologies --n-questions 20
"""
import argparse
import json
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
import networkx as nx

from topologies.definitions import get_few_topologies, get_all_topologies, HYBRID_ROLES
from mas.agent import Agent, AgentConfig
from mas.runner import MASRunner
from mas.prompts import parse_answer
from metrics.graph_metrics import TopologyMetrics


def build_agents(graph: nx.DiGraph, topo_name: str, model: str, stub: bool) -> dict:
    """Builds agents with correct roles for any topology."""
    agent_nodes = [n for n in graph.nodes if n != "task"]

    if topo_name in HYBRID_ROLES:
        role_map = HYBRID_ROLES[topo_name]
    else:
        order = [n for n in nx.topological_sort(graph) if n != "task"]
        roles_cycle = ["solver", "critic", "aggregator"]
        role_map = {n: roles_cycle[i % len(roles_cycle)] for i, n in enumerate(order)}

    return {
        node: Agent(AgentConfig(
            agent_id=node,
            role=role_map.get(node, "solver"),
            model=model,
            stub=stub,
        ))
        for node in agent_nodes
    }


def main():
    for p in [Path(__file__).parent.parent, Path(__file__).parent]:
        if (p / ".env").exists():
            load_dotenv(p / ".env", override=True)
            break

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--stub", action="store_true", help="Use stub agents (no API calls)")
    parser.add_argument("--all-topologies", action="store_true", help="Run all 18 topologies")
    args = parser.parse_args()

    print(f"Loading {args.n_questions} GSM8K questions...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.n_questions}]")
    except Exception:
        import urllib.request, json as _json
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        print("  HuggingFace unavailable, downloading from GitHub...")
        with urllib.request.urlopen(url) as r:
            lines = r.read().decode().splitlines()
        ds = [_json.loads(l) for l in lines[: args.n_questions]]

    topologies = get_all_topologies() if args.all_topologies else get_few_topologies(n_agents=3)
    metrics_engine = TopologyMetrics()

    print(f"\nTopologies: {len(topologies)} | stub={args.stub} | questions={len(ds)}\n")

    results = []
    for topo_name, graph in topologies.items():
        agents = build_agents(graph, topo_name, args.model, args.stub)
        runner = MASRunner(graph, agents)
        metrics = metrics_engine.compute(graph, topo_name)

        correct = 0
        for i, item in enumerate(ds):
            question = item["question"]
            gt = int(re.search(r"####\s*(-?\d+)", item["answer"]).group(1))
            answer_text = runner.run(question)
            pred = parse_answer(answer_text)
            if pred is not None and pred == gt:
                correct += 1
            if args.stub and i == 0:
                print(f"  [{topo_name}] sample: {answer_text[:100]!r}")

        accuracy = correct / len(ds)
        print(f"{topo_name:30s}  accuracy={accuracy:.2f}  ({correct}/{len(ds)})")

        results.append({
            "topology": topo_name,
            "accuracy": accuracy,
            "metrics": {
                "diameter":           metrics.diameter,
                "avg_degree":         metrics.avg_degree,
                "structural_entropy": metrics.structural_entropy,
                "spectral_gap":       metrics.spectral_gap,
                "task_centrality":    metrics.task_centrality,
            },
        })

    Path("results").mkdir(exist_ok=True)
    out = Path("results/results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
