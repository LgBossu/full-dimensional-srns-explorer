from abc import ABC, abstractmethod
from itertools import permutations

import numpy as np
from behaviors import routed_index_to_indices, routed_indices_to_index
from loguru import logger


class FullDimEquation:
    """
    This class acts as an immutable wrapper around a numpy array,
    providing a structured way to handle equations with multiple coefficients.
    """

    def __init__(self, coefficients: np.ndarray) -> None:
        """
        Initialize an FullDimEquation object with the given coefficients.

        :param coefficients: A numpy array representing the coefficients of the equation.
        """
        if not isinstance(coefficients, np.ndarray):
            raise TypeError("Coefficients must be a numpy array.")
        # if not np.issubdtype(coefficients.dtype, np.number):
        #     raise TypeError("Coefficients array must have a numeric dtype.")
        if not coefficients.flags["WRITEABLE"]:
            pass  # Already immutable
        else:
            coefficients.setflags(write=False)
        self._coefficients = coefficients
        self._shape = self._coefficients.shape
        self._ndim = self._coefficients.ndim

    def __repr__(self) -> str:
        return f"FullDimEquation(coefficients={self._coefficients})"

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
    def flatten(self) -> "FullDimEquation":
        """
        Flatten the coefficients of the equation to a 1D numpy array.

        :return: A 1D numpy array of coefficients.
        """
        flattened_coefficients = self._coefficients.flatten()
        return FullDimEquation(flattened_coefficients)

    def reshape(self, new_shape: tuple[int, ...]) -> "FullDimEquation":
        """
        Reshape the coefficients of the equation to a new shape.

        :param new_shape: A tuple representing the new shape.
        :return: A new FullDimEquation object with reshaped coefficients.
        """
        reshaped_coefficients = self._coefficients.reshape(new_shape)
        return FullDimEquation(reshaped_coefficients)

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

    def project_on(self, other: "FullDimEquation") -> "FullDimEquation":
        """
        Project this equation onto another equation.

        :param other: Another FullDimEquation object to project onto.
        :return: A new FullDimEquation object representing the projection.

        This projection method is useful to determine linear relationships
        between equations, allowing for checks of consistency across coefficients.
        """
        if not isinstance(other, FullDimEquation):
            raise TypeError("Other must be an instance of FullDimEquation.")
        if self._shape != other.shape:
            raise ValueError("Both equations must have the same shape for projection.")

        norm_other = other.norm()
        if norm_other == 0:
            raise ValueError("The other equation cannot be a zero vector.")

        projection = (
            np.dot(self._coefficients.flatten(), other.coefficients.flatten()) / norm_other**2
        ) * other.coefficients
        return FullDimEquation(projection)

    def __eq__(self, other: object) -> bool:
        """
        Check if two FullDimEquation objects are equal.

        :param other: Another FullDimEquation object to compare with.
        :return: True if the equations are equal, False otherwise.
        """
        if not isinstance(other, FullDimEquation):
            return False
        return np.array_equal(self._coefficients, other.coefficients)

    def approx_equal(self, other: "FullDimEquation", tol: float = 1e-10) -> bool:
        """
        Check if two FullDimEquation objects are approximately equal within a tolerance.

        :param other: Another FullDimEquation object to compare with.
        :param tol: The tolerance for comparison.
        :return: True if the equations are approximately equal, False otherwise.
        """
        if not isinstance(other, FullDimEquation):
            return False
        return np.allclose(self._coefficients, other.coefficients, atol=tol)


class FullDimEquationsQuotienter:
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
            # logger.debug(
            #     "Inequalities should be a numpy array, not a list. Converting to numpy array."
            # )
            ineq = np.array(ineq)

        assert isinstance(ineq, np.ndarray), "Inequalities must be a numpy array."
        assert len(ineq.shape) == 2, "Inequalities must be a 2D numpy array."
        if ineq.shape[1] != 2 * delta**2 * m**2 + 1:
            logger.error(
                f"Expected inequalities to have measured vectors' shape (n, {2 * delta**2 * m**2 + 1}), "  # noqa: E501
                f"but got {ineq.shape}. This may lead to unexpected behavior."
            )

        self._ineq = ineq

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
        if self._ineq.shape[1] != expected_dim:
            raise ValueError(
                f"Expected inequalities to have dimension {expected_dim}, "
                f"but got {self._ineq.shape[1]}."
            )

    # Properties
    @property
    def ineq(self) -> np.ndarray:
        """
        Get the inequalities as a numpy array.

        :return: The inequalities as a numpy array.
        """
        return self._ineq.copy()

    # Checker for expected dimension and behavior
    def check_compatible_dimension(
        self,
        equation: FullDimEquation,
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
        equation_a: FullDimEquation,
        equation_b: FullDimEquation,
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
        equation: FullDimEquation,
    ) -> FullDimEquation:
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
        shaped_equation: FullDimEquation,
    ) -> FullDimEquation:
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
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of a values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, list(perm), :, :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [
            self.flatten_equation(FullDimEquation(eq)) for eq in permuted_equations
        ]

        return permuted_equations

    def permute_b(
        self,
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of b values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, list(perm), :, :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [
            self.flatten_equation(FullDimEquation(eq)) for eq in permuted_equations
        ]

        return permuted_equations

    def permute_x(
        self,
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of x values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, :, list(perm), :])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [
            self.flatten_equation(FullDimEquation(eq)) for eq in permuted_equations
        ]

        return permuted_equations

    def permute_y(
        self,
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
        """Apply to an equation the permutation of coordinates
        corresponding to relabelings of y values."""
        permuted_equations = []
        permutable = self.prepare_equation(equation)

        for perm in permutations(range(self.delta)):
            permuted_equations.append(permutable.coefficients[:, :, :, :, list(perm)])

        # Flatten the permuted equations back to the original shape
        permuted_equations = [
            self.flatten_equation(FullDimEquation(eq)) for eq in permuted_equations
        ]

        return permuted_equations

    def permute(
        self,
        axis: int | str,
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
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

    def extend_non_redundant(
        self,
        base_list: list[FullDimEquation],
        new_equations: list[FullDimEquation],
    ) -> None:
        """
        Extend a list of equations with new equations, ensuring no duplicates.

        This method checks if each new equation is equivalent to any existing equation
        in the base list. If not, it adds the new equation to the base list.

        Modifies the base_list in place.
        """
        for new_eq in new_equations:
            # logger.debug(
            #     f"Checking if new equation {new_eq.coefficients} is redundant in list {len(base_list)}."  # noqa: E501
            # )
            if not any(self.are_equivalent(base_eq, new_eq) for base_eq in base_list):
                # logger.debug(f"Adding new equation {new_eq.coefficients} to the base list.")
                base_list.append(new_eq)

        return None

    def gen_orbit(
        self,
        equation: FullDimEquation,
    ) -> list[FullDimEquation]:
        """
        Generate all orbit equations for a given equation.

        This method generates all possible permutations of the equation's coefficients
        across the specified axes, returning a list of FullDimEquation objects representing
        each unique permutation.
        """
        orbit = []
        axes_to_permute = [[bool(i) for i in bin(bin_string).zfill(4)] for bin_string in range(16)]
        for axes in axes_to_permute:
            if not any(axes):
                # If no axes are selected, skip this iteration
                continue
            if axes[0]:
                permuted = self.permute("a", equation)
                self.extend_non_redundant(orbit, permuted)
            if axes[1]:
                permuted = self.permute("b", equation)
                self.extend_non_redundant(orbit, permuted)
            if axes[2]:
                permuted = self.permute("x", equation)
                self.extend_non_redundant(orbit, permuted)
            if axes[3]:
                permuted = self.permute("y", equation)
                self.extend_non_redundant(orbit, permuted)

        return orbit


class Translator(ABC):
    """
    Abstract base class for translating equations into different representations.
    """

    pass


class BinaryTranslator(Translator):
    """
    Translator for converting equations to and from correlator representation,
    in the binary input-output case.
    """

    def __init__(self) -> None:
        """
        Initialize the BinaryTranslator.
        """
        self.equation_size = 2 * 2**2 * 2**2 + 1  # For delta=2, m=2
        self.vector_size = 2 * 2**2 * 2**2  # For delta=2, m=2

        self.correlator_size = 2 + (2 * 2) + (2 * 2 * 2)

        super().__init__()

    def _indices_to_coord(
        self,
        indices: tuple[int, int, int, int, int] | list[int],
    ) -> int:
        """
        Converts from the a,b,x,y,z indexation to vector index.
        :param indices: A tuple or list of indices (a, b, x, y, z).
        :return: The corresponding vector index.
        """
        if isinstance(indices, list):
            assert len(indices) == 5, "Indices must be a list of length 5."

        return routed_indices_to_index(
            indices[0],
            indices[1],
            indices[2],
            indices[3],
            indices[4],
            delta=2,
            m=2,
        )

    def _coord_to_indices(self, coord: int) -> tuple[int, int, int, int, int]:
        """
        Converts from vector index to the a,b,x,y,z indexation.
        :param coord: The vector index.
        :return: A tuple of indices (a, b, x, y, z).
        """
        res = routed_index_to_indices(coord, delta=2, m=2)
        return res

    def to_correlator(self, equation: FullDimEquation) -> np.ndarray:
        """
        Convert a FullDimEquation to a correlator representation.

        :param equation: The FullDimEquation to convert.
        :return: A numpy array representing the correlator.
        """
        if equation.size != self.equation_size:
            raise ValueError(
                f"Equation size must be {self.equation_size}, but got {equation.size}."
            )

        equation = equation.flatten()

        # Get the constant part and the coefficients part
        cst_part = equation.coefficients[0]
        coeff_part = equation.coefficients[1:]

        # Initialize the correlator array
        correlator_vec = np.zeros(self.correlator_size, dtype=DummyText)

        # Compute the correlator values
        """For this specific part, we can apply to the equation coefficients the same equation
        as we would to a regular behavior vector, given that evaluating the equation
        is just taking the dot product of the coefficients with the behavior vector."""
        ab_values = [(a, b) for a in range(2) for b in range(2)]

        # A naive first implementation : we assume that such correlators are well-defined
        cur_idx: int = 0
        for x in range(2):
            # <A_x> correlators
            for a, b in ab_values:
                correlator_vec[cur_idx] += (-1) ** (a) * coeff_part[
                    self._indices_to_coord((a, b, x, 0, 0))
                ]
            cur_idx += 1
        for y, z in [(y, z) for y in range(2) for z in range(2)]:
            # <B_yz> correlators
            for a, b in ab_values:
                correlator_vec[cur_idx] += (-1) ** (b) * coeff_part[
                    self._indices_to_coord((a, b, 0, y, z))
                ]
            cur_idx += 1
        for x, y, z in [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]:
            # <A_x B_yz> correlators
            for a, b in ab_values:
                correlator_vec[cur_idx] += (-1) ** (a + b) * coeff_part[
                    self._indices_to_coord((a, b, x, y, z))
                ]
            cur_idx += 1

        # Set the constant part
        res = np.hstack((cst_part, correlator_vec))
        return res


class DummyText:
    """Text that supports left multiplication so that we can pass it through the translator,
    and check the resulting symbolic expression."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"DummyText({self.text})"

    def __str__(self) -> str:
        return self.text

    def __mul__(self, other: "DummyText") -> "DummyText":
        return DummyText(f"{self.text}*{str(other)}")

    def __rmul__(self, other: "DummyText") -> "DummyText":
        if other == 0:
            return DummyText("")
        if other == 1:
            return DummyText(self.text)
        if other == -1:
            return DummyText(f"-{self.text}")
        return DummyText(f"{str(other)}*{self.text}")

    def __add__(self, other: "DummyText") -> "DummyText":
        return DummyText(f"{self.text} + {str(other)}")

    def __radd__(self, other: "DummyText") -> "DummyText":
        return DummyText(f"{str(other)} + {self.text}")


symbolic_equation = np.array(
    [DummyText("cst")]  # Constant term
    + [
        DummyText("p({}{}|{}{}{})".format(a, b, x, y, z))
        for z in "01"
        for a in "01"
        for b in "01"
        for x in "01"
        for y in "01"
    ],
    dtype=DummyText,
)


if __name__ == "__main__":
    equation_file = "output/measured_h_representation_inequality_delta_2_m_2.csv"

    with open(equation_file, "r") as f:
        # Read the coefficients from the file
        equations_list = np.loadtxt(f, delimiter=",", skiprows=1)
    equations = [FullDimEquation(coefficients) for coefficients in equations_list]

    translator = BinaryTranslator()
    # for eq in equations:
    #     correlator = translator.to_correlator(eq)
    #     print(f"Correlator: {correlator}")

    # Example of using the translator with symbolic equations
    symbolic_eq = FullDimEquation(symbolic_equation)
    # print(f"Symbolic equation: {symbolic_eq.coefficients}")
    correlator = translator.to_correlator(symbolic_eq)
    # print(f"Symbolic correlator: {str(correlator).replace("+ -", "- ")}")

    cur = -1
    for line in correlator:
        line = str(line).replace("+ -", "- ")
        if cur < 0:
            print("Constant term: ", line)
        elif cur < 2:
            print("<A_x>      = ", line)
        elif cur < 6:
            print("<B_yz>     = ", line)
        else:
            print("<A_x B_yz> = ", line)
        cur += 1
    print(f"Total correlator size: {len(correlator)}")

    # Let us take an equation and see how it applies to the coordinates
    for i, eq in enumerate(equations):
        print(f"Ineq {i+1:2}: {np.dot(eq.coefficients, symbolic_equation)}")
