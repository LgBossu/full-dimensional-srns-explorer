"""This module aims to provide in matrix form the equations defining the no signaling set."""

# TODO : MIGHT NEED REFACTORING AFTER BEHAVIORS REFACTORING

from abc import ABC, abstractmethod

import behaviors
import numpy as np
from loguru import logger
from scipy.linalg import svd


class BehaviorSet(ABC):
    """
    Abstract class for defining sets of behaviors.
    """

    def __init__(self, delta: int, m: int, routed: bool = True, positivity: bool = True):
        """
        Initialize the behavior set.
        :param delta: Number of outcomes for each measurement.
        :param m: Number of measurements.
        :param routed: Whether the set is viewed in the routed setting.
        :param positivity: Whether vectors of the set verify v >= 0.
        """

        if not isinstance(delta, int) or not delta > 0:
            raise ValueError("Delta must be a positive integer.")
        if not isinstance(m, int) or not m > 0:
            raise ValueError("m must be a positive integer.")
        self.delta = delta
        self.m = m
        self.routed = routed  # Whether the set is viewed in the routed setting
        self.positivity = positivity  # Whether vectors of the set verify v >= 0

    @abstractmethod
    def get_equations(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the equations defining the set. Equations formalism may vary depending on the set,
        as each is known differently and thus tests for membership may differ.
        """
        pass

    def get_dimension(self, tolerance: float = 1e-10) -> int:
        """
        Get the dimension of the set, computed as the rank of the equations matrix,
        i.e. the number of independent equations.
        """
        # The dimension of the set is the number of independent equations
        equations, _ = self.get_equations()

        _, s, _ = svd(equations)
        # Count the number of singular values greater than the tolerance
        rank = np.sum(s > tolerance)

        return rank

    def get_reduced_equations(self, tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the reduced equations defining the set.
        """
        equations, right_side = self.get_equations()
        U, s, _ = svd(equations)
        rank = np.sum(s > tolerance)
        reduced_equations = U[:, :rank].T @ equations
        reduced_right_side = U[:, :rank].T @ right_side

        return reduced_equations, reduced_right_side


class NoSignalingSet(BehaviorSet):
    def __init__(self, delta: int, m: int, routed: bool = True):
        if not routed:
            logger.warning(
                "This class implements methods in the routed setting only. Please fall back to a dedicated class for non-routed cases if available."  # noqa: E501
            )
            logger.warning("Class instantiation will forcibly set routed to True.")

        super().__init__(delta, m)

    def get_equations(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the no-signaling equations for the routed case.
        """
        # Polytope dimension
        # dim = (2 * (delta - 1) * m) + ((delta - 1) ** 2 * m**2)
        # logger.trace(f"Polytope dimension: {dim}")

        # Initialize the equations
        equations = []
        right_side = []

        # Normalization equations
        for x, y, z in [(i, j, k) for i in range(self.m) for j in range(self.m) for k in [0, 1]]:
            eq = np.zeros(2 * self.delta**2 * self.m**2)
            for a, b in [(i, j) for i in range(self.delta) for j in range(self.delta)]:
                eq[behaviors.routed_indices_to_index(a, b, x, y, z, self.delta, self.m)] = 1
            equations.append(eq)
            right_side.append(1)
            logger.trace(f"Normalization equation: {eq} = 1, xyz = {x, y, z}")

        # No-signaling equations
        # From Alice to Bob
        for b, y, z in [
            (i, j, k) for i in range(self.delta) for j in range(self.m) for k in [0, 1]
        ]:
            for x in range(self.m):
                eq = np.zeros(2 * self.delta**2 * self.m**2)
                if x == 0:
                    continue
                else:
                    for a in range(self.delta):
                        eq[
                            behaviors.routed_indices_to_index(a, b, x, y, z, self.delta, self.m)
                        ] = -1
                        eq[behaviors.routed_indices_to_index(a, b, 0, y, z, self.delta, self.m)] = 1
                    equations.append(eq)
                    right_side.append(0)
                    logger.trace(f"No-signaling equation: {eq} = 0, bxyz = {b, x, y, z}")
        # From Bob to Alice
        for a, x in [(i, j) for i in range(self.delta) for j in range(self.m)]:
            for y, z in [(j, k) for j in range(self.m) for k in [0, 1]]:
                eq = np.zeros(2 * self.delta**2 * self.m**2)
                if (y, z) == (0, 0):
                    continue
                else:
                    for b in range(self.delta):
                        eq[
                            behaviors.routed_indices_to_index(a, b, x, y, z, self.delta, self.m)
                        ] = -1
                        eq[behaviors.routed_indices_to_index(a, b, x, 0, 0, self.delta, self.m)] = 1
                    equations.append(eq)
                    right_side.append(0)
                    logger.trace(f"No-signaling equation: {eq} = 0, axyz = {a, x, y, z}")

        # Convert to numpy arrays
        equations = np.array(equations)
        right_side = np.array(right_side)

        # TODO : reduce the rank of the equations matrix to equal dim(B) - dim(NS) or smth like that

        # Log the shapes and values of the equations and right side
        logger.trace(f"Delta: {self.delta}, m: {self.m}")
        logger.trace(f"Equations shape: {equations.shape}")
        logger.trace(f"Equations: {equations}")
        logger.trace(f"Right side shape: {right_side.shape}")
        logger.trace(f"Right side: {right_side}")

        return equations, right_side


class ShortRangeNoSignalingSet(BehaviorSet):
    def __init__(self, delta: int, m: int, measured_behavior: behaviors.RoutedBehavior):
        super().__init__(delta, m)

        assert (
            self.delta == measured_behavior.delta
        ), "Declared delta does not match measured behavior delta"
        assert self.m == measured_behavior.m, "Declared m does not match measured behavior m"

    def express_as_function_of_q(self) -> np.ndarray:
        """
        Returns the matrix M which,
        given a vector q of coordinates (q(ab|xy), q(a beta|x)),
        will return the corresponding short-range no-signaling behavior p as:
        p = f(q) = M @ q
        """
        # Dimension of the short-path q vector
        dim_q_s = self.delta**2 * self.m**2
        # Dimension of the long-path q vector
        dim_q_L = self.m * self.delta ** (self.m + 1)
        # Dimension of q
        dim_q = dim_q_s + dim_q_L
        # Dimension of p, the classical routed behavior dim
        dim_p = 2 * self.delta**2 * self.m**2

        # Initialize the matrix M
        M = np.zeros((dim_p, dim_q))

        # The first dim_q_s block enforces q_s = p(z=S)
        M[:dim_q_s, :dim_q_s] = np.eye(dim_q_s)

        # The second dim_q_L block enforces p(z=L) = sum_beta,beta_y=b q(a beta|x)
        lacking_betas = [
            list(np.base_repr(i, self.delta).rjust(self.m, "0"))
            for i in range(self.delta ** (self.m - 1))
        ]  # Lacking betas holds the list of all beta tuples, with one missing coordinate
        # The missing coordinate is the one to be inserted with 'beta_y=b'
        logger.debug(
            f"Estimated memory complexity of lacking_betas: {len(lacking_betas) * 28} bytes"
        )

        for line_idx in range(self.delta**2 * self.m**2, dim_p):
            # We loop over all the lines of the second block,
            # i.e. coordinates of p(z=L)
            a, b, x, y, _ = behaviors.routed_index_to_indices(line_idx, delta=self.delta, m=self.m)

            for beta in [tuple(l_beta[:y] + [b] + l_beta[y:]) for l_beta in lacking_betas]:
                M[
                    line_idx,
                    behaviors.routed_indices_to_index((a, beta, x), delta=self.delta, m=self.m),
                ] = 1
                # TODO : if needed, there is room for optimization here :
                # do not call short_range_index_to_indices for each beta,
                # but recompute the array of indices once with array operations
                # and update M[line_idx, computed_arr]=1)

        return M

    def express_with_irreductible_ns_constraint(self) -> np.ndarray:
        """
        Enforce the no-signaling equation for sum_beta = q(a|x)
        below the matrix M, s.t.
        (p, 0) = (M, mat_ns_beta) @ q

        For each (a,x), we take as reference a short-path q(a|x)
        and enforce that the long-path sum over beta holds the equality.
        Although short-path no-signaling of q
        (i.e. sum_b q(ab|xyS)=q(a|x))
        will eventually be given when assuming p=f(q),
        enforcing LP-no-signaling on short-path coordinates
        ensures consistency across values of z,
        and ease of implementation.
        """
        M = self.express_as_function_of_q()

        # Dimension of q
        dim_q = (self.delta**2 * self.m**2) + (self.m * self.delta ** (self.m + 1))
        NS_enforcer = np.zeros((self.delta * self.m, dim_q))
        values_of_beta = [
            tuple(np.base_repr(i, self.delta).rjust(self.m, "0")) for i in range(self.delta**self.m)
        ]
        # There are exactly delta*m equations to enforce in the above form
        for line_idx, (a, x) in enumerate(
            [(i, j) for i in range(self.delta) for j in range(self.m)]
        ):
            # TODO : may add tqdm in such places to check progress in real time for greater dims
            row_eq = np.zeros(dim_q)
            for b, y in [(i, j) for i in range(self.delta) for j in range(self.m)]:
                # Given a,x,z=S, we loop over all b,y coordinates
                # to compute the reference q(a|x)
                row_eq[
                    behaviors.short_range_indices_to_index(
                        (a, b, x, y, 0), delta=self.delta, m=self.m
                    )
                ] = 1
            for beta in values_of_beta:
                row_eq[
                    behaviors.short_range_index_to_indices(
                        (a, beta, x, 1), delta=self.delta, m=self.m
                    )
                ] = -1
            NS_enforcer[line_idx] = row_eq

        return np.vstack((M, NS_enforcer))

    def get_equations(
        self,
        measured_behavior: behaviors.RoutedBehavior,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assuming a parameter vector of the form x = (alpha, q(ab|xy), q(a beta|x)),
        returns the matrices enforcing the test equation :
        alpha * (1 - p) + f(q) = 1   &&   q is beta-no-signaling

        1 denotes the maximally mixed state over (delta, m) routed Bell experiments.
        Integrating problem constants into the output matrices,
        it reformulates the above equations as :
        A @ x = b
        where the first rows of A and b correspond to the first equation (equality),
        and the other rows correspond to the second equation (no-signaling).
        """
        assert (
            self.delta == measured_behavior.delta
        ), "Declared delta does not match measured behavior delta"
        assert self.m == measured_behavior.m, "Declared m does not match measured behavior m"

        base_matrix = self.express_with_irreductible_ns_constraint()
        maximally_mixed_state = (1 / self.delta**2) * np.ones(2 * self.delta**2 * self.m**2)

        n_last_rows = base_matrix.shape[0] - (2 * self.delta**2 * self.m**2)
        filler_vector = np.zeros(n_last_rows)

        first_A_column = np.vstack(
            (maximally_mixed_state - measured_behavior.get_vector()).reshape(-1, 1),
            filler_vector.reshape(-1, 1),
        )

        A = np.hstack((first_A_column, base_matrix))
        b = np.concatenate((maximally_mixed_state, filler_vector))

        return A, b


def routed_no_signaling_equations(delta: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the no-signaling equations for the routed case.
    """
    logger.warning(
        "This function is deprecated. Use NoSignalingSet.routed_no_signaling_equations instead, or a dedicated class."  # noqa: E501
    )
    raise NotImplementedError("Deprecated. Turn to dedicated classes instead.")
