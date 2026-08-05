"""Shared helpers for scripts/analyze.py and scripts/stats_analysis.py."""
from typing import Dict


def n_agents_map() -> Dict[str, int]:
    """Number of agents (non-'task' nodes) for every registered topology.

    Looked up against topologies.definitions.get_all_topologies() — callers
    must pass topology names that exist in that registry.
    """
    from topologies.definitions import get_all_topologies
    return {
        name: sum(1 for node in g.nodes if node != "task")
        for name, g in get_all_topologies().items()
    }


def lookup_n_agents(topology_names, agents_map: Dict[str, int] = None):
    """Returns [n_agents_map()[name] for name in topology_names], raising
    KeyError with the offending name if a topology isn't in the registry."""
    agents_map = agents_map if agents_map is not None else n_agents_map()
    out = []
    for name in topology_names:
        if name not in agents_map:
            raise KeyError(name)
        out.append(agents_map[name])
    return out
