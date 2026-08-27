from __future__ import annotations

import numpy as np
from tkf_models.recon.specs import HierarchySpec, TemporalSpec


def build_summing_matrix(all_nodes: list[str], bottom_nodes: list[str], structure: dict[str, list[str]]) -> np.ndarray:
    """Constructs the summing matrix S of shape (n_total_series, m_bottom_series).

    such that y = S @ y_bottom
    """
    n = len(all_nodes)
    m = len(bottom_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    bottom_to_idx = {node: i for i, node in enumerate(bottom_nodes)}

    S = np.zeros((n, m), dtype=np.float64)

    # Helper function to find all bottom descendants of a given node
    def get_bottom_descendants(node: str) -> list[str]:
        if node in bottom_to_idx:
            return [node]
        children = structure.get(node, [])
        descendants = []
        for child in children:
            descendants.extend(get_bottom_descendants(child))
        return descendants

    for node in all_nodes:
        row_idx = node_to_idx[node]
        descendants = get_bottom_descendants(node)
        for b_node in descendants:
            col_idx = bottom_to_idx[b_node]
            S[row_idx, col_idx] += 1.0

    return S


def build_temporal_matrix(k_factors: list[int]) -> np.ndarray:
    """Builds temporal summing matrix K for multi-frequency hierarchies."""
    total_bottom = k_factors[0]  # finest frequency
    rows = []
    for k in k_factors:
        n_blocks = total_bottom // k
        for b in range(n_blocks):
            row = np.zeros(total_bottom)
            row[b * k : (b + 1) * k] = 1.0
            rows.append(row)
    return np.vstack(rows)
