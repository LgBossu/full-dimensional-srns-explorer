from itertools import permutations

import numpy as np
from loguru import logger


class QuotientInequalities:
    def __init__(
        self,
        delta: int,
        m: int,
        ineq: list[np.ndarray] | np.ndarray,
        lin_set: set[int],
    ) -> None:
        self.delta = delta
        self.m = m

        if isinstance(ineq, list):
            logger.debug(
                "Inequalities should be a numpy array, not a list. Converting to numpy array."
            )
            ineq = np.array(ineq)

        assert isinstance(ineq, np.ndarray), "Inequalities must be a numpy array."
        assert len(ineq.shape) == 2, "Inequalities must be a 2D numpy array."
        if ineq.shape[1] != 2 * delta**2 * m**2 + 1:
            logger.error(
                f"Expected inequalities to have measured vectors' shape (n, {2 * delta**2 * m**2 + 1}), "  # noqa: E501
                f"but got {ineq.shape}. This may lead to unexpected behavior."
            )

        self.ineq = ineq

        assert isinstance(lin_set, set), "Linear set must be a set of integers."
        assert all(isinstance(i, int) for i in lin_set), "Linear set must contain integers."
        self.lin_set = lin_set

    # # SIGN CONSISTENCY
    # def canonicalize(eq: np.ndarray) -> np.ndarray:
    #     """
    #     Canonicalizes a hypeperplane equation by flattening its array if needed,
    #     and setting the first nonzero element to be positive.

    #     This enables consistent representation of hyperplanes, for later
    #     comparison in sets.

    #     Note that :
    #     - This may switch the sign of the hyperplane equation (ie, the normal vector's direction).
    #     - This does not guarantee that the hyperplane is normalized (ie, unit length).
    #     - This returns a 1D numpy array.
    #     """
    #     flat = eq.flatten()
    #     idx = np.flatnonzero(flat)
    #     if idx.size and flat[idx[0]] < 0:
    #         flat = -flat
    #     return flat

    # # FORMATTING FOR PERMUTATIONS
    # def format_hyperplane(
    #     self,
    #     equation: np.ndarray,
    # ) -> tuple[np.ndarray]:
    #     # TODO : ensure the function is applied to canonicalized hyperplanes.

    #     el_S, el_L, el_NS = (
    #         equation[0 : self.delta**2 * self.m**2],
    #         equation[self.delta**2 * self.m**2 : 2 * self.delta**2 * self.m**2],
    #         equation[2 * self.delta**2 * self.m**2 :],
    #     )
    #     el_S = el_S.reshape((self.delta, self.delta, self.m, self.m))
    #     el_L = el_L.reshape((self.delta, self.delta, self.m, self.m))
    #     el_NS = el_NS.reshape((self.delta,) * self.m)
    #     return (el_S, el_L, el_NS)

    # def format_list_of_hyperplanes(
    #     self,
    #     equations: list[list[np.ndarray]],
    # ) -> list[tuple[np.ndarray]]:
    #     formatted: list[tuple[np.ndarray]] = []
    #     for el in equations:
    #         formatted.append(self.format_hyperplane(el))

    #     return formatted

    # def flatten_hyperplane(
    #     self,
    #     hyperplane: tuple[np.ndarray, np.ndarray, np.ndarray],
    # ) -> np.ndarray:
    #     el_S, el_L, el_NS = hyperplane

    #     equation = np.concatenate((el_S.flatten(), el_L.flatten(), el_NS.flatten()))

    #     return equation

    # def flatten_list_of_hyperplanes(
    #     self,
    #     hyperplanes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    # ) -> np.ndarray:
    #     flattened = []
    #     for hyperplane in hyperplanes:
    #         flattened.append(self.flatten_hyperplane(hyperplane))
    #     return np.array(flattened, dtype=int)

    # PERMUTATIONS
    def permute_a(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of a values."""
        permuted_equations = []

        pass  # TODO

        return permuted_equations

    def permute_b(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of b values."""
        permuted_equations = []

        pass  # TODO

        return permuted_equations

    def permute_x(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of x values."""
        permuted_equations = []

        pass  # TODO

        return permuted_equations

    def permute_y(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of y values."""
        permuted_equations = []

        pass  # TODO

        return permuted_equations

    def permute(
        self,
        axis: int | str,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Permute an equation along the specified axis."""

        list_of_permutations = [
            self.permute_a,
            self.permute_b,
            self.permute_x,
            self.permute_y,
        ]

        if isinstance(axis, str):
            axis = axis.lower()
            if axis == "a":
                permuted_equations = self.permute_a(equation)
            elif axis == "b":
                permuted_equations = self.permute_b(equation)
            elif axis == "x":
                permuted_equations = self.permute_x(equation)
            elif axis == "y":
                permuted_equations = self.permute_y(equation)
            else:
                raise ValueError(f"Unknown axis: {axis}")
        elif isinstance(axis, int):
            if 0 <= axis < 4:
                permuted_equations = list_of_permutations[axis](equation)
            else:
                raise ValueError(f"Axis index out of range: {axis}")
        else:
            raise TypeError(f"Axis must be an int or a str, got {type(axis)}")

        return permuted_equations
