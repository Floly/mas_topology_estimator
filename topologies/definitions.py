import networkx as nx
from typing import Dict


def get_topologies(n_agents: int = 3) -> Dict[str, nx.DiGraph]:
    """
    Returns {name: DiGraph} with 4 topologies.
    Nodes: "task" (virtual input), "A0".."A{n-1}".
    Edges: directed, from source to receiver.
    """
    return {
        "chain": _chain(n_agents),
        "star": _star(n_agents),
        "full": _full(n_agents),
        "random": _random(n_agents),
    }


def _chain(n: int) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = ["task"] + [f"A{i}" for i in range(n)]
    G.add_nodes_from(nodes)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G


def _star(n: int) -> nx.DiGraph:
    """
    task → each agent independently, then all agents → aggregator node A{n}.
    A{n} is the aggregator that sees all agent outputs.
    """
    G = nx.DiGraph()
    agents = [f"A{i}" for i in range(n)]
    aggregator = f"A{n}"
    G.add_nodes_from(["task"] + agents + [aggregator])
    for a in agents:
        G.add_edge("task", a)
        G.add_edge(a, aggregator)
    return G


def _full(n: int) -> nx.DiGraph:
    """
    Full DAG: edges from task to all agents, and between every pair
    of agents (only forward: Ai → Aj where i < j). No back-edges to task.
    """
    G = nx.DiGraph()
    agents = [f"A{i}" for i in range(n)]
    G.add_nodes_from(["task"] + agents)
    for a in agents:
        G.add_edge("task", a)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(f"A{i}", f"A{j}")
    return G


def _random(n: int, seed: int = 42) -> nx.DiGraph:
    """
    Random DAG with n+1 nodes (task + agents) that is fully connected.
    Generates gnm graph, removes back-edges to make it a DAG,
    then ensures every node is reachable from 'task'.
    """
    import random
    rng = random.Random(seed)

    nodes = ["task"] + [f"A{i}" for i in range(n)]
    node_idx = {v: i for i, v in enumerate(nodes)}

    # Generate random edges respecting topological order (guaranteed DAG)
    # Use ~n+2 edges for reasonable connectivity
    n_nodes = len(nodes)
    possible = [(nodes[i], nodes[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(possible)
    chosen = possible[: n + 2]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen)

    # Ensure full reachability from "task": add missing edges
    for node in nodes[1:]:
        if not nx.has_path(G, "task", node):
            # connect via any already-reachable predecessor candidate
            reachable = {v for v in nodes if nx.has_path(G, "task", v) and node_idx[v] < node_idx[node]}
            src = rng.choice(list(reachable)) if reachable else "task"
            G.add_edge(src, node)

    assert nx.is_directed_acyclic_graph(G), "random topology is not a DAG"
    assert all(nx.has_path(G, "task", node) for node in nodes[1:]), "not fully reachable"

    return G
