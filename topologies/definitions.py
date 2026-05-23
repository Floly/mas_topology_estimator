import networkx as nx
from typing import Dict

def hierarchical(n: int = 3) -> nx.DiGraph:
    """task → manager → [A0, A1, ..., A{n-1}]
    Двухуровневое дерево: менеджер получает задачу,
    раздаёт подзадачи исполнителям.
    Специальная роль: "manager" (промпт нужно добавить в prompts.py).
    """
    G = nx.DiGraph()
    workers = [f"A{i}" for i in range(n)]
    G.add_nodes_from(["task", "manager"] + workers)
    G.add_edge("task", "manager")
    for w in workers:
        G.add_edge("manager", w)
    assert nx.is_directed_acyclic_graph(G)
    return G
 
 
def debate() -> nx.DiGraph:
    """task → debater0, task → debater1 → judge
       debater0 → judge, debater1 → judge
    Два агента независимо решают задачу, судья выбирает лучший ответ.
    Специальные роли: "debater0", "debater1", "judge".
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "debater0", "debater1", "judge"])
    G.add_edge("task", "debater0")
    G.add_edge("task", "debater1")
    G.add_edge("debater0", "judge")
    G.add_edge("debater1", "judge")
    assert nx.is_directed_acyclic_graph(G)
    return G
 
 
def pipeline_with_critic() -> nx.DiGraph:
    """task → solver → critic → aggregator
             task --------↗ (skip-connection: critic видит task напрямую)
    Критик видит и решение solver-а, и исходную задачу.
    Специальные роли: "solver", "critic", "aggregator".
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "solver", "critic", "aggregator"])
    G.add_edge("task", "solver")
    G.add_edge("solver", "critic")
    G.add_edge("task", "critic")       # skip-connection
    G.add_edge("critic", "aggregator")
    assert nx.is_directed_acyclic_graph(G)
    return G
 
 
def two_layer_ensemble(n: int = 2) -> nx.DiGraph:
    """task → solver0, task → solver1, ..., → aggregator
              task -----------------↗ (skip-connection к aggregator)
    Параллельные решатели + агрегатор, который видит и task напрямую.
    Специальные роли: "solver0", "solver1", ..., "aggregator".
    """
    G = nx.DiGraph()
    solvers = [f"solver{i}" for i in range(n)]
    G.add_nodes_from(["task"] + solvers + ["aggregator"])
    for s in solvers:
        G.add_edge("task", s)
        G.add_edge(s, "aggregator")
    G.add_edge("task", "aggregator")   # skip-connection
    assert nx.is_directed_acyclic_graph(G)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Parametric topology builders  (public)
# ─────────────────────────────────────────────────────────────────────────────

def chain(n: int) -> nx.DiGraph:
    """task → A0 → A1 → ... → A{n-1}"""
    G = nx.DiGraph()
    nodes = ["task"] + [f"A{i}" for i in range(n)]
    G.add_nodes_from(nodes)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    return G


def star(n: int) -> nx.DiGraph:
    """task → A0, task → A1, ..., task → A{n-1}"""
    G = nx.DiGraph()
    G.add_nodes_from(["task"] + [f"A{i}" for i in range(n)])
    for i in range(n):
        G.add_edge("task", f"A{i}")
    return G


def fc(n: int) -> nx.DiGraph:
    """task → all agents, Ai → Aj for all i < j"""
    G = nx.DiGraph()
    agents = [f"A{i}" for i in range(n)]
    G.add_nodes_from(["task"] + agents)
    for a in agents:
        G.add_edge("task", a)
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(agents[i], agents[j])
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid topology builders
# ─────────────────────────────────────────────────────────────────────────────

def star_then_chain() -> nx.DiGraph:
    """
    task → {A0, A1, A2}  (parallel star)
    A0 → A3 → A4         (chain from one branch)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "A0", "A1", "A2", "A3", "A4"])
    for a in ["A0", "A1", "A2"]:
        G.add_edge("task", a)
    G.add_edge("A0", "A3")
    G.add_edge("A3", "A4")
    return G


def chain_with_star_sink() -> nx.DiGraph:
    """
    task → A0 → A1 → A2  (chain backbone)
    task, A0, A1, A2 → A3  (A3 is the shared aggregator)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "A0", "A1", "A2", "A3"])
    G.add_edge("task", "A0")
    G.add_edge("A0", "A1")
    G.add_edge("A1", "A2")
    for src in ["task", "A0", "A1", "A2"]:
        G.add_edge(src, "A3")
    return G


def fc_then_sink() -> nx.DiGraph:
    """
    fc(3) first layer, then A0,A1,A2 → A3 (final aggregator)
    """
    G = fc(3)
    G.add_node("A3")
    for a in ["A0", "A1", "A2"]:
        G.add_edge(a, "A3")
    return G


def star_with_chain_backbone() -> nx.DiGraph:
    """
    task → A0 → A1 → A2  (chain backbone)
    task → A1, task → A2  (skip-connections)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "A0", "A1", "A2"])
    G.add_edge("task", "A0")
    G.add_edge("A0", "A1")
    G.add_edge("A1", "A2")
    G.add_edge("task", "A1")
    G.add_edge("task", "A2")
    return G


def two_stars_merged() -> nx.DiGraph:
    """
    task → {A0, A1, A2, A3}
    A1 → A4, A3 → A4  (two branches merge at aggregator A4)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "A0", "A1", "A2", "A3", "A4"])
    for a in ["A0", "A1", "A2", "A3"]:
        G.add_edge("task", a)
    G.add_edge("A1", "A4")
    G.add_edge("A3", "A4")
    return G


def chain_of_stars() -> nx.DiGraph:
    """
    task → A0
    A0 → {A1, A2}   (fan-out)
    {A1, A2} → A3   (fan-in)
    A3 → {A4, A5}   (fan-out)
    """
    G = nx.DiGraph()
    G.add_nodes_from(["task", "A0", "A1", "A2", "A3", "A4", "A5"])
    G.add_edge("task", "A0")
    G.add_edge("A0", "A1")
    G.add_edge("A0", "A2")
    G.add_edge("A1", "A3")
    G.add_edge("A2", "A3")
    G.add_edge("A3", "A4")
    G.add_edge("A3", "A5")
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Role mappings for hybrid topologies
# ─────────────────────────────────────────────────────────────────────────────

HYBRID_ROLES: Dict[str, Dict[str, str]] = {
    "star_then_chain":          {"A0": "solver", "A1": "solver", "A2": "solver", "A3": "critic", "A4": "aggregator"},
    "chain_with_star_sink":     {"A0": "solver", "A1": "solver", "A2": "solver", "A3": "aggregator"},
    "fc_then_sink":             {"A0": "solver", "A1": "solver", "A2": "solver", "A3": "aggregator"},
    "star_with_chain_backbone": {"A0": "solver", "A1": "critic", "A2": "aggregator"},
    "two_stars_merged":         {"A0": "solver", "A1": "solver", "A2": "solver", "A3": "solver", "A4": "aggregator"},
    "chain_of_stars":           {"A0": "solver", "A1": "critic", "A2": "critic", "A3": "critic", "A4": "aggregator", "A5": "aggregator"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────────────────────

def get_all_topologies() -> Dict[str, nx.DiGraph]:
    """
    Returns all 18 topologies:
      Parametric: chain/star/fc × n in {3,4,5} → 12
      Hybrid: 6
    All validated as DAGs with full reachability from 'task'.
    """
    topos: Dict[str, nx.DiGraph] = {}

    for n in [3, 4, 5]:
        topos[f"chain_{n}"] = chain(n)
        topos[f"star_{n}"]  = star(n)
        topos[f"fc_{n}"]    = fc(n)
        topos[f"hierarchical_{n}"] = hierarchical(n)
        topos[f"two_layer_ensemble_{n}"] = two_layer_ensemble(n)

    topos["star_then_chain"]          = star_then_chain()
    topos["chain_with_star_sink"]     = chain_with_star_sink()
    topos["fc_then_sink"]             = fc_then_sink()
    topos["star_with_chain_backbone"] = star_with_chain_backbone()
    topos["two_stars_merged"]         = two_stars_merged()
    topos["chain_of_stars"]           = chain_of_stars()
    topos["debate"]                   = debate()
    topos["pipeline_with_critic"]     = pipeline_with_critic()

    for name, G in topos.items():
        assert nx.is_directed_acyclic_graph(G), f"{name} is not a DAG"
        for node in G.nodes:
            if node == "task":
                continue
            assert nx.has_path(G, "task", node), \
                f"{name}: node {node} unreachable from task"

    return topos


def get_few_topologies(n_agents: int = 3) -> Dict[str, nx.DiGraph]:
    """Original 4 topologies — kept for backward compatibility with run_poc.py."""
    return {
        "chain":  chain(n_agents),
        "star":   star(n_agents),
        "full":   fc(n_agents),
        "random": random(n_agents),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Original private helpers  (used only by get_topologies)
# ─────────────────────────────────────────────────────────────────────────────


def random(n: int, seed: int = 42) -> nx.DiGraph:
    import random
    rng = random.Random(seed)

    nodes = ["task"] + [f"A{i}" for i in range(n)]
    node_idx = {v: i for i, v in enumerate(nodes)}

    n_nodes = len(nodes)
    possible = [(nodes[i], nodes[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(possible)
    chosen = possible[: n + 2]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(chosen)

    for node in nodes[1:]:
        if not nx.has_path(G, "task", node):
            reachable = {v for v in nodes if nx.has_path(G, "task", v) and node_idx[v] < node_idx[node]}
            src = rng.choice(list(reachable)) if reachable else "task"
            G.add_edge(src, node)

    assert nx.is_directed_acyclic_graph(G), "random topology is not a DAG"
    assert all(nx.has_path(G, "task", node) for node in nodes[1:]), "not fully reachable"
    return G
