import networkx as nx
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricsResult:
    name: str
    diameter: float
    avg_degree: float
    structural_entropy: float
    spectral_gap: float
    task_centrality: float


class TopologyMetrics:
    def compute(self, graph: nx.DiGraph, name: str) -> MetricsResult:
        ug = graph.to_undirected()

        try:
            diam = nx.average_shortest_path_length(ug)
        except nx.NetworkXError:
            paths = []
            for src in ug.nodes:
                lengths = nx.single_source_shortest_path_length(ug, src)
                paths.extend(lengths.values())
            diam = float(np.mean(paths)) if paths else float("inf")

        avg_deg = float(np.mean([d for _, d in graph.out_degree()]))

        degrees = np.array([d for _, d in graph.degree()])
        vol = degrees.sum()
        probs = degrees / vol if vol > 0 else degrees
        probs = probs[probs > 0]
        se = -float(np.sum(probs * np.log2(probs)))

        L = nx.normalized_laplacian_matrix(ug).toarray()
        eigvals = np.sort(np.linalg.eigvalsh(L))
        gap = float(eigvals[1]) if len(eigvals) > 1 else 0.0

        bc = nx.betweenness_centrality(graph)
        task_c = bc.get("task", 0.0)

        return MetricsResult(
            name=name,
            diameter=diam,
            avg_degree=avg_deg,
            structural_entropy=se,
            spectral_gap=gap,
            task_centrality=task_c,
        )
