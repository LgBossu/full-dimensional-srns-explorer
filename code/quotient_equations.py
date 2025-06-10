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

    def check_compatible_dimension_self(self) -> None:
        """
        Check if the dimension of the inequalities matches the expected dimension.

        Certain methods in this class require the inequalities to have a specific
        dimension. This check ensures that no errors occur later in the code due
        to dimension mismatches.
        """
        expected_dim = 2 * self.delta**2 * self.m**2 + 1
        if self.ineq.shape[1] != expected_dim:
            raise ValueError(
                f"Expected inequalities to have dimension {expected_dim}, "
                f"but got {self.ineq.shape[1]}."
            )

    def check_compatible_dimension(
        self,
        equation: np.ndarray,
    ) -> None:
        """
        Check if the dimension of the given equation matches the expected dimension.

        Like `check_compatible_dimension_self`, this method ensures that the
        equation has the correct dimension before proceeding with further
        operations.
        Notably, when reshaping equations for permutations and the like, this
        check is crucial to avoid errors due to dimension mismatches.
        """
        expected_dim = 2 * self.delta**2 * self.m**2
        n_coordinates = len(equation.flatten())
        if (n_coordinates) != expected_dim:
            raise ValueError(
                f"Expected equation to have dimension {expected_dim}, " f"but got {n_coordinates}."
            )

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
    def prepare_equation(
        self,
        equation: np.ndarray,
    ) -> np.ndarray:
        """
        Format an equation to be used in permutations.

        Checks if the equation has the correct dimension beforehand,
        and if no errors are raised, returns the equation
        reshaped in the expected format to facilitate coordinates permutations.
        """
        self.check_compatible_dimension(equation)

        reformatted_equation = equation.reshape((2, self.delta, self.delta, self.m, self.m))

        return reformatted_equation

    def flatten_equation(
        self,
        equation: np.ndarray,
    ) -> np.ndarray:
        """
        Flatten an equation to a 1D numpy array.

        This is the inverse operation of `prepare_equation`.
        """
        self.check_compatible_dimension(equation)

        flattened_equation = equation.flatten()

        return flattened_equation

    # PERMUTATIONS
    def permute_a(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of a values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable[:, list(perm), :, :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(eq) for eq in permuted_equations]

        return permuted_equations

    def permute_b(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of b values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable[:, :, list(perm), :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(eq) for eq in permuted_equations]

        return permuted_equations

    def permute_x(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of x values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable[:, :, :, list(perm), :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(eq) for eq in permuted_equations]

        return permuted_equations

    def permute_y(
        self,
        equation: np.ndarray,
    ) -> list[np.ndarray]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of y values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable[:, :, :, :, list(perm)])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(eq) for eq in permuted_equations]

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
