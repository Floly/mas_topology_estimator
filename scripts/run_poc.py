"""
Runs the full PoC:
  1. Loads GSM8K questions
  2. For each of 4 topologies runs MAS
  3. Computes graph metrics
  4. Saves results/results.json

Usage:
    python scripts/run_poc.py --model gpt-3.5-turbo --n-questions 20
    python scripts/run_poc.py --stub --n-questions 5   # no API calls
"""
import argparse
import json
import re
import sys
from pathlib import Path

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset

from topologies.definitions import get_topologies
from mas.agent import Agent, AgentConfig
from mas.runner import MASRunner
from mas.prompts import assign_roles, parse_answer
from metrics.graph_metrics import TopologyMetrics


def _build_agents(topo_name: str, graph, roles, model: str, stub: bool):
    agent_nodes = [n for n in graph.nodes if n != "task"]
    return {
        node: Agent(AgentConfig(
            agent_id=node,
            role=roles.get(node, "solver"),
            model=model,
            stub=stub,
        ))
        for node in agent_nodes
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-3.5-turbo")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--stub", action="store_true",
                        help="Use stub agents (no API calls) for pipeline testing")
    args = parser.parse_args()

    print(f"Loading {args.n_questions} GSM8K questions...")
    try:
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.n_questions}]")
    except Exception:
        # fallback: raw GitHub file
        import urllib.request, json as _json
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        print("  HuggingFace unavailable, downloading from GitHub...")
        with urllib.request.urlopen(url) as r:
            lines = r.read().decode().splitlines()
        ds = [_json.loads(l) for l in lines[: args.n_questions]]

    topologies = get_topologies(n_agents=3)
    metrics_engine = TopologyMetrics()
    roles = assign_roles(4)  # star topology adds A3 as aggregator

    results = []
    for topo_name, graph in topologies.items():
        agents = _build_agents(topo_name, graph, roles, args.model, args.stub)
        runner = MASRunner(graph, agents)
        metrics = metrics_engine.compute(graph, topo_name)

        print(f"\n[{topo_name}] nodes={list(graph.nodes)} edges={list(graph.edges)}")
        if args.stub:
            print(f"  stub mode — no API calls")

        correct = 0
        for i, item in enumerate(ds):
            question = item["question"]
            gt_match = re.search(r"####\s*(-?\d+)", item["answer"])
            gt = int(gt_match.group(1))
            answer_text = runner.run(question)
            pred = parse_answer(answer_text)
            if pred is not None and pred == gt:
                correct += 1
            if args.stub and i == 0:
                print(f"  sample answer_text: {answer_text[:120]!r}")

        accuracy = correct / len(ds)
        print(f"  accuracy = {accuracy:.2f}  ({correct}/{len(ds)})")

        results.append({
            "topology": topo_name,
            "accuracy": accuracy,
            "metrics": {
                "diameter": metrics.diameter,
                "avg_degree": metrics.avg_degree,
                "structural_entropy": metrics.structural_entropy,
                "spectral_gap": metrics.spectral_gap,
                "task_centrality": metrics.task_centrality,
            },
        })

    Path("results").mkdir(exist_ok=True)
    out = Path("results/results.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
