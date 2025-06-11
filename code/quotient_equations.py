from itertools import permutations

import numpy as np
from loguru import logger


class Equation:
    """
    This class acts as an immutable wrapper around a numpy array,
    providing a structured way to handle equations with multiple coefficients.
    """

    def __init__(self, coefficients: np.ndarray) -> None:
        """
        Initialize an Equation object with the given coefficients.

        :param coefficients: A numpy array representing the coefficients of the equation.
        """
        if not isinstance(coefficients, np.ndarray) or coefficients.ndim != 1:
            raise TypeError("Coefficients must be a 1D numpy array.")
        if not np.issubdtype(coefficients.dtype, np.number):
            raise TypeError("Coefficients array must have a numeric dtype.")
        if not coefficients.flags["WRITEABLE"]:
            pass  # Already immutable
        else:
            coefficients.setflags(write=False)
        self._coefficients = coefficients
        self._shape = self._coefficients.shape
        self._ndim = self._coefficients.ndim

    def __repr__(self) -> str:
        return f"Equation(coefficients={self._coefficients})"

    # Getters for private attributes
    @property
    def coefficients(self) -> np.ndarray:
        """
        Get the coefficients of the equation.

        :return: The coefficients as a numpy array.
        """
        return self._coefficients.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the coefficients of the equation.

        :return: The shape of the coefficients as a tuple.
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions of the coefficients of the equation.

        :return: The number of dimensions.
        """
        return self._ndim

    @property
    def size(self) -> int:
        """
        Get the total number of coefficients in the equation.

        :return: The total number of coefficients.
        """
        return self._coefficients.size

    # Numpy interface and others
    def flatten(self) -> "Equation":
        """
        Flatten the coefficients of the equation to a 1D numpy array.

        :return: A 1D numpy array of coefficients.
        """
        flattened_coefficients = self._coefficients.flatten()
        return Equation(flattened_coefficients)

    def reshape(self, new_shape: tuple[int, ...]) -> "Equation":
        """
        Reshape the coefficients of the equation to a new shape.

        :param new_shape: A tuple representing the new shape.
        :return: A new Equation object with reshaped coefficients.
        """
        reshaped_coefficients = self._coefficients.reshape(new_shape)
        return Equation(reshaped_coefficients)

    def __len__(self) -> int:
        """
        Get the number of coefficients in the equation.

        :return: The number of coefficients.

        WARNING : This method returns the total number of coefficients,
        overriding the default behavior of len() for numpy arrays.
        This is useful for understanding the size of the equation,
        thus this specific implementation.
        """
        return self.size

    def norm(self) -> float:
        """
        Compute the norm of the coefficients of the equation.

        :return: The norm of the coefficients.
        """
        return float(np.linalg.norm(self._coefficients))

    def project_on(self, other: "Equation") -> "Equation":
        """
        Project this equation onto another equation.

        :param other: Another Equation object to project onto.
        :return: A new Equation object representing the projection.

        This projection method is useful to determine linear relationships
        between equations, allowing for checks of consistency across coefficients.
        """
        if not isinstance(other, Equation):
            raise TypeError("Other must be an instance of Equation.")
        if self._shape != other.shape:
            raise ValueError("Both equations must have the same shape for projection.")

        norm_other = other.norm()
        if norm_other == 0:
            raise ValueError("The other equation cannot be a zero vector.")

        projection = (
            np.dot(self._coefficients.flatten(), other.coefficients.flatten()) / norm_other**2
        ) * other.coefficients
        return Equation(projection)

    def __eq__(self, other: object) -> bool:
        """
        Check if two Equation objects are equal.

        :param other: Another Equation object to compare with.
        :return: True if the equations are equal, False otherwise.
        """
        if not isinstance(other, Equation):
            return False
        return np.array_equal(self._coefficients, other.coefficients)

    def approx_equal(self, other: "Equation", tol: float = 1e-10) -> bool:
        """
        Check if two Equation objects are approximately equal within a tolerance.

        :param other: Another Equation object to compare with.
        :param tol: The tolerance for comparison.
        :return: True if the equations are approximately equal, False otherwise.
        """
        if not isinstance(other, Equation):
            return False
        return np.allclose(self._coefficients, other.coefficients, atol=tol)


class QuotientEquations:
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
        equation: Equation,
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

    # CONSISTENCY ACROSS COEFFICIENTS
    def are_equivalent(
        self,
        equation_a: Equation,
        equation_b: Equation,
        ineq: bool = False,
        tol_nonzero: float = 1e-10,
    ) -> bool:
        """
        Check if two equations are equivalent.

        This method checks if two equations reflect the same linear
        relationship, regardless of the sign or magnitude of the coefficients.

        We actually check if the vectors are colinear,
        and, if `ineq` is True, we also check that they have the same direction
        (i.e., the same sign).
        """
        proj_a = equation_a.project_on(equation_b)
        self_similar = proj_a.approx_equal(equation_a)

        if self_similar and ineq:
            # We need to check that the direction is right
            a = equation_a.coefficients
            b = equation_b.coefficients
            nonzero_mask = (np.abs(a) > tol_nonzero) & (np.abs(b) > tol_nonzero)
            return bool(np.all(np.sign(a[nonzero_mask]) == np.sign(b[nonzero_mask])))

        return self_similar

    # # FORMATTING FOR PERMUTATIONS
    def prepare_equation(
        self,
        equation: Equation,
    ) -> Equation:
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
        shaped_equation: Equation,
    ) -> Equation:
        """
        Flatten an equation to a 1D numpy array.

        This is the inverse operation of `prepare_equation`.
        """
        self.check_compatible_dimension(shaped_equation)

        flattened_equation = shaped_equation.flatten()

        return flattened_equation

    # PERMUTATIONS
    def permute_a(
        self,
        equation: Equation,
    ) -> list[Equation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of a values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, list(perm), :, :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(Equation(eq)) for eq in permuted_equations]

        return permuted_equations

    def permute_b(
        self,
        equation: Equation,
    ) -> list[Equation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of b values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, list(perm), :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(Equation(eq)) for eq in permuted_equations]

        return permuted_equations

    def permute_x(
        self,
        equation: Equation,
    ) -> list[Equation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of x values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, :, list(perm), :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(Equation(eq)) for eq in permuted_equations]

        return permuted_equations

    def permute_y(
        self,
        equation: Equation,
    ) -> list[Equation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of y values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, :, :, list(perm)])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [self.flatten_equation(Equation(eq)) for eq in permuted_equations]

        return permuted_equations

    def permute(
        self,
        axis: int | str,
        equation: Equation,
    ) -> list[Equation]:
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
