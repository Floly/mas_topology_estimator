import networkx as nx
from typing import Dict, Tuple

from mas.agent import Agent


class MASRunner:
    def __init__(self, graph: nx.DiGraph, agents: Dict[str, Agent]):
        self.graph = graph
        self.agents = agents

    def run(self, task: str) -> Tuple[str, int]:
        """Returns (final_answer, total_tokens_used_this_call)."""
        order = list(nx.topological_sort(self.graph))
        messages: Dict[str, str] = {"task": task}

        tokens_before = {node: agent.total_tokens for node, agent in self.agents.items()}

        for node in order:
            if node == "task":
                continue
            preds = list(self.graph.predecessors(node))
            incoming = [messages[p] for p in preds if p in messages]
            messages[node] = self.agents[node].run(task, incoming)

        # last agent node in topological order is the final answer
        agent_nodes = [n for n in order if n != "task"]
        tokens_used = sum(
            self.agents[n].total_tokens - tokens_before[n]
            for n in agent_nodes
        )
        return messages[agent_nodes[-1]], tokens_used
