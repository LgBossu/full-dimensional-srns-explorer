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
        Initialize the set with delta and m parameters.
        :param delta: The number of possible outputs for Alice and Bob
        :param m: The number of possible inputs for Alice and Bob
        :param routed: Whether the set is viewed in the routed setting (default: True)
        :param positivity: Whether the set is positive (default: True)
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
        Get the equations defining the set.
        """
        pass

    def get_dimension(self, tolerance: float = 1e-10) -> int:
        """
        Get the dimension of the set, computed as the rank of the equations matrix,
        i.e. the number of independent equations.
        """
        # The dimension of the set is the number of independent equations
        equations, _ = self.get_equations()

        U, s, Vh = svd(equations)
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
    def __init__(self, delta: int, m: int, measured_behavior: behaviors.Behavior):
        super().__init__(delta, m)

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

        return M

    def get_equations(self) -> tuple[np.ndarray, np.ndarray]:
        pass


def routed_no_signaling_equations(delta: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the no-signaling equations for the routed case.
    """
    logger.warning(
        "This function is deprecated. Use NoSignalingSet.routed_no_signaling_equations instead, or a dedicated class."  # noqa: E501
    )
    raise NotImplementedError("Deprecated. Turn to dedicated classes instead.")
