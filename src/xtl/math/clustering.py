import numpy as np


def find_maximal_cliques(adjacency: np.ndarray) -> list[list[int]]:
    """
    Find all maximal cliques in a graph using Bron-Kerbosch with pivoting.

    :param adjacency: Square adjacency matrix where True indicates an edge.
    :return: Maximal cliques as sorted vertex indices, ordered by descending size.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.asarray(adjacency, dtype=bool)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError('`adjacency` must be a square 2D matrix')

    maximal_cliques: list[list[int]] = []
    largest_clique_size = 0

    def bron_kerbosch(
        current_clique: list[int],
        candidate_vertices: list[int],
        excluded_vertices: list[int],
    ) -> None:
        """
        Recursive function implementing the Bron-Kerbosch algorithm with pivoting to find maximal cliques.
        """
        nonlocal largest_clique_size

        if not candidate_vertices and not excluded_vertices:
            clique = sorted(current_clique)
            maximal_cliques.append(clique)
            largest_clique_size = max(largest_clique_size, len(clique))
            return

        # Prune branches that cannot beat the best clique found so far.
        if maximal_cliques and len(current_clique) + len(candidate_vertices) <= largest_clique_size:
            return

        pivot_pool = candidate_vertices + excluded_vertices
        pivot = max(
            pivot_pool,
            key=lambda vertex: np.sum(adjacency[vertex][candidate_vertices]),
        )
        extension_candidates = [
            vertex for vertex in candidate_vertices if not adjacency[pivot, vertex]
        ]

        for vertex in extension_candidates:
            bron_kerbosch(
                current_clique + [vertex],
                [neighbor for neighbor in candidate_vertices if adjacency[vertex, neighbor]],
                [neighbor for neighbor in excluded_vertices if adjacency[vertex, neighbor]],
            )
            candidate_vertices.remove(vertex)
            excluded_vertices.append(vertex)

    bron_kerbosch([], list(range(len(adjacency))), [])
    return sorted(maximal_cliques, key=len, reverse=True)
