"""
Given a set of points, use linear programming to prune points that lie
in the convex hull of the other points.

This algorithm optimises by pruning points that do not contribute to the convex hull,
on the fly, from the working set. For non-trivial sets (where non-extremal points are present),
this can reduce the complexity from the naive O(n^2), for a tasty advantage.
"""

from typing import List, Set, Tuple

import numpy as np
from loguru import logger
from scipy.optimize import linprog
from tqdm import tqdm


class PruneForVertices:
    """
    Class to prune non-vertex points from a set of points.
    """

    def _round_points(self, decimals: int) -> None:
        logger.info(f"Rounding points to {decimals} decimals for numerical stability.")
        self._points = np.round(self._points, decimals)

    def _deduplicate_points(self) -> None:
        """
        Remove duplicate points from the set.
        """
        logger.info("Removing duplicate points from the set.")
        self._points = np.unique(self._points, axis=0)

    def __init__(self, points: List[np.ndarray]) -> None:
        """
        Initialize with a list of points.
        Each point is expected to be a numpy array.
        """
        self._points: np.ndarray = np.array(points, dtype=float)
        assert self._points.ndim == 2, "Points should properly cast to a 2D array."

        self._round_points(decimals=10)
        self._deduplicate_points()

        self._working_points = self._points.copy()

        self.total_unique_inputs = self._points.shape[0]

        logger.info(f"Initialized with {self.total_unique_inputs} unique points.")

    def __iter__(self):
        """
        Iterate over the points in the set.
        """
        for point in self._points:
            yield point

    def determine_vertex(self, point_index: int, tol: float = 1e-9) -> bool:
        """
        Determine if a point is a vertex of the convex hull.
        A point is a vertex if it cannot be expressed as a convex combination
        of other points in the set.
        """
        others = np.delete(self._working_points, point_index, axis=0)  # shape (n-1, d)
        A = np.vstack((np.ones((1, others.shape[0])), others.T))  # shape (d+1, n-1)
        b = np.hstack((1, self._working_points[point_index]))  # shape (d+1,)

        # Linear problem : A.x = b, x>=0
        res = linprog(
            c=np.zeros(A.shape[1]),
            A_eq=A,
            b_eq=b,
            bounds=(0, 1),
            method="highs",
        )
        return not res.success  # The point is a vertex if the LP fails to find a solution.

    def prune(self) -> Tuple[Set[int], Set[int], int]:
        """
        Prune the points that are not vertices of the convex hull.

        Returns:
            - A set of indices of the vertices.
            - A set of indices of the non-vertex points.
            - The total number of points in the initial set.
        """
        all_indices = set(range(self._points.shape[0]))
        vertex_indices = set()
        total = self._points.shape[0]

        current_index = 0

        for i, point in enumerate(
            tqdm(self, colour="yellow", desc="Pruning points", total=self.total_unique_inputs)
        ):
            if self.determine_vertex(current_index):
                vertex_indices.add(i)
                current_index += 1  # We go to the next line in the working set
            else:
                # If the point is not a vertex, remove it from the working set
                self._working_points = np.delete(self._working_points, current_index, axis=0)
                # We do not increment current_index here, as we just removed a point :
                # the next point's index was shifted down by one, and thus is still at current_index

        logger.success(f"Done! Pruned {len(all_indices) - len(vertex_indices)} non-vertex points.")
        # The remaining points in _working_points are the vertices
        return vertex_indices, all_indices - vertex_indices, total


if __name__ == "__main__":
    file_name = "boxworld_extremals_sanitycheck.txt"

    vertices = []
    seen_vertices = set()

    logger.info(f"Loading vertices from {file_name}...")
    with open(file_name, "r") as f:
        for line in f:
            if line.strip() and line.strip() not in seen_vertices:
                vertex = np.fromstring(line.strip(), sep=",", dtype=float)
                vertices.append(vertex)
                seen_vertices.add(line.strip())

    pruner = PruneForVertices(vertices)
    vertex_indices, non_vertex_indices, total_points = pruner.prune()

    logger.info(f"Total points: {total_points}")
    logger.info(f"Vertex points: {len(vertex_indices)}")
    logger.info(f"Non-vertex points: {len(non_vertex_indices)}")

    with open("pruned_vertices_222.txt", "w") as f:
        for index in vertex_indices:
            f.write(",".join(map(str, pruner._points[index])) + "\n")
