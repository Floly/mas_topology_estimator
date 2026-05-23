import networkx as nx
from typing import Dict

from mas.agent import Agent


class MASRunner:
    def __init__(self, graph: nx.DiGraph, agents: Dict[str, Agent]):
        self.graph = graph
        self.agents = agents

    def run(self, task: str) -> str:
        order = list(nx.topological_sort(self.graph))
        messages: Dict[str, str] = {"task": task}

        for node in order:
            if node == "task":
                continue
            preds = list(self.graph.predecessors(node))
            incoming = [messages[p] for p in preds if p in messages]
            messages[node] = self.agents[node].run(task, incoming)

        # last agent node in topological order is the final answer
        agent_nodes = [n for n in order if n != "task"]
        return messages[agent_nodes[-1]]
