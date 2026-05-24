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
    flow_entropy: float
    von_neumann_entropy: float


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

        # Flow Entropy
        A_mat = nx.to_numpy_array(ug)  # adjacency (undirected for stationary dist)
        deg_vec = A_mat.sum(axis=1)
        deg_vec_safe = np.where(deg_vec > 0, deg_vec, 1)
        P = A_mat / deg_vec_safe[:, None]  # row-stochastic transition matrix

        # Stationary distribution (uniform for regular, otherwise left eigenvector)
        # Use degree-proportional approximation: π_i = deg_i / vol
        vol_rw = deg_vec.sum()
        pi = deg_vec / vol_rw if vol_rw > 0 else np.ones(len(deg_vec)) / len(deg_vec)

        # H_flow = -sum_i π_i * sum_j P_ij * log2(P_ij)
        row_entropy = np.array([
            -np.sum(P[i] * np.log2(np.where(P[i] > 0, P[i], 1)))
            for i in range(len(P))
        ])
        flow_entropy = float(np.dot(pi, row_entropy))
        
        # Spectral entropy of normalized Laplacian eigenvalues (λ1=0 excluded)
        nonzero_eigvals = eigvals[1:]
        tr_L = nonzero_eigvals.sum()
        if tr_L > 0:
            probs_vn = nonzero_eigvals / tr_L
            probs_vn = probs_vn[probs_vn > 0]
            von_neumann_entropy = -float(np.sum(probs_vn * np.log2(probs_vn)))
        else:
            von_neumann_entropy = 0.0

        return MetricsResult(
            name=name,
            diameter=diam,
            avg_degree=avg_deg,
            structural_entropy=se,
            spectral_gap=gap,
            task_centrality=task_c,
            flow_entropy=flow_entropy,
            von_neumann_entropy=von_neumann_entropy,
        )
