from abc import ABC, abstractmethod

import numpy as np
from loguru import logger


class Behavior(ABC):
    """
    Abstract class for defining behaviors.
    """

    # CONSTRUCTOR
    def __init__(self, delta: int, m: int, vector: np.ndarray = None):
        """
        Initialize the behavior with delta and m parameters.
        :param delta: The number of possible outputs for Alice and Bob
        :param m: The number of possible inputs for Alice and Bob
        """
        if not isinstance(delta, int) or not delta > 0:
            raise ValueError("Delta must be a positive integer.")
        if not isinstance(m, int) or not m > 0:
            raise ValueError("m must be a positive integer.")
        self.delta = delta
        self.m = m
        self.behavior_vector = vector
        self.vector_shape: tuple[int] = None  # The dimension of the latent space

    # VECTOR AND MATRIX FORMS CONVERSIONS
    @abstractmethod
    def behavior_vector_to_matrix(self, behavior_vector):
        """
        Convert a vector behavior to its matrix representation
        :param behavior_vector: The behavior vector
        :return: The behavior matrix
        """
        pass

    @abstractmethod
    def behavior_matrix_to_vector(self, behavior_matrix):
        """
        Convert a matrix behavior to its vector representation
        :param behavior_matrix: The behavior matrix
        :return: The behavior vector
        """
        pass

    # GETTERS
    def get_vector(self):
        return self.behavior_vector

    def get_matrix(self):
        return self.behavior_vector_to_matrix(self.behavior_vector)

    def get_delta(self):
        return self.delta

    def get_m(self):
        return self.m

    def get_vector_shape(self):
        return self.vector_shape

    # STRING REPRESENTATIONS
    def __repr__(self):
        return f"Behavior({self.behavior_vector})"

    @abstractmethod
    def __str__(self):
        pass

    # EQUALITY
    @abstractmethod
    def compare_array(self, other):
        """
        Determine if the other object can be compared to this behavior
        """
        pass

    def __eq__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return np.array_equal(self.behavior_vector, comparable.behavior_vector)
        return False

    def __ne__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return not np.array_equal(self.behavior_vector, comparable.behavior_vector)
        return True

    def __lt__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return np.all(self.behavior_vector < comparable.behavior_vector)
        return False

    def __le__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return np.all(self.behavior_vector <= comparable.behavior_vector)
        return False

    def __gt__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return np.all(self.behavior_vector > comparable.behavior_vector)
        return False

    def __ge__(self, other):
        comparable = self.compare_array(other)
        if comparable:
            return np.all(self.behavior_vector >= comparable.behavior_vector)
        return False

    # MATRIX OPERATIONS
    def __matmul__(self, other):
        if not isinstance(other, np.ndarray):
            raise ValueError("The right operand must be a numpy array.")
        return self.get_vector() @ other

    def __rmatmul__(self, other):
        if not isinstance(other, np.ndarray):
            raise ValueError("The left operand must be a numpy array.")
        return other @ self.get_vector()

    # CONDITIONS
    def positivity(self):
        return self >= 0

    @abstractmethod
    def normalization(self, atol=1e-10):
        """
        Check if the behavior is normalized
        """
        pass

    @abstractmethod
    def no_signaling(self, atol=1e-10):
        """
        Check if the behavior is no-signaling
        """
        pass

    def is_normalized(self, atol=1e-10):
        return self.positivity() and self.normalization(atol=atol)

    def is_no_signaling(self, atol=1e-10):
        return self.positivity() and self.normalization(atol=atol) and self.no_signaling(atol=atol)


class RoutedBehavior(Behavior):
    """
    Class to represent a behavior measured in the routed experiment setting,
    assimilated to a p(ab|xyz) distribution.
    """

    def __init__(self, delta: int, m: int, vector: np.ndarray = None):
        """
        Initialize the behavior with delta and m parameters.
        :param delta: The number of possible outputs for Alice and Bob
        :param m: The number of possible inputs for Alice and Bob
        """
        super().__init__(delta, m, vector)

        self.vector_shape = (2 * delta**2 * m**2,)
        assert (
            self.behavior_vector.shape == (2 * delta**2 * m**2,)
        ), f"Invalid shape {self.behavior_vector.shape}. Expected {self.vector_shape}, to match declared values (delta={self.delta}, m={self.m})."  # noqa: E501

        self.matrix_shape = (2, delta**2, m**2)

    def behavior_vector_to_matrix(self, behavior_vector):
        return np.reshape(behavior_vector, self.matrix_shape)

    def behavior_matrix_to_vector(self, behavior_matrix):
        return np.reshape(behavior_matrix, self.vector_shape)

    def __str__(self):
        matrix_form = self.behavior_vector_to_matrix(self.behavior_vector)
        res: str = "Behavior:\n"
        res += f"Short path (z=S):\n{matrix_form[0]}\n"
        res += f"Long path (z=L) :\n{matrix_form[1]}\n"
        res += "-" * 12
        return res

    def compare_array(self, other):
        logger.trace(f"Checking if variable {other} can be treated as a Behavior for comparisons")

        if isinstance(other, RoutedBehavior):
            return other
        elif isinstance(other, np.ndarray):
            if other.shape == self.vector_shape:
                return RoutedBehavior(self.delta, self.m, other)
            elif other.shape == self.matrix_shape:
                return RoutedBehavior(self.delta, self.m, self.behavior_matrix_to_vector(other))
            logger.warning(f"Invalid shape on conversion into Behavior: {other.shape}")
        elif isinstance(other, int) or isinstance(other, float):
            return RoutedBehavior(self.delta, self.m, np.ones(self.vector_shape) * other)

        logger.warning(
            f"Attempted to convert {other} into Behavior for comparison, but it is not a valid type"
        )
        return None

    def normalization(self, atol=1e-10):
        """
        Check if the behavior is normalized
        """
        matrix = self.get_matrix()
        return np.all(abs(np.sum(matrix, axis=1) - 1) < atol)

    def no_signaling(self, atol=1e-10):
        summable = np.reshape(self.behavior_vector, (2, self.delta, self.delta, self.m, self.m))

        logger.trace(f"Summable array shape: {summable.shape}")
        logger.trace(f"Summable array: {summable}")
        sum_over_a: np.ndarray = np.sum(summable, axis=1)
        logger.trace(f"Shape of summed array: {sum_over_a.shape}")
        logger.trace(f"Summed array: {sum_over_a}")
        sum_over_a = np.moveaxis(sum_over_a, 2, -1)
        # logger.trace(f"Transposed summed array shape: {sum_over_a.shape}")
        logger.trace(f"Reordered summed array: {sum_over_a}")
        sum_over_a = sum_over_a.reshape(2 * self.delta * self.m, self.m)
        logger.trace(f"Reshaped summed array shape: {sum_over_a.shape}")
        logger.trace(f"Reshaped summed array: {sum_over_a}")
        a_checksum = np.abs(sum_over_a - sum_over_a[:, [0]]) < atol
        a_no_signaling = np.all(a_checksum, axis=1)

        sum_over_b: np.ndarray = np.sum(summable, axis=2)
        sum_over_b = np.moveaxis(sum_over_b, 0, -1)
        sum_over_b = sum_over_b.reshape(self.delta * self.m, 2 * self.m)
        b_checksum = np.abs(sum_over_b - sum_over_b[:, [0]]) < atol
        b_no_signaling = np.all(b_checksum, axis=1)

        return np.all(a_no_signaling) and np.all(b_no_signaling)


# COORDINATE UTILS
# Given a 1-D index, return the corresponding a,b,x,y,z indices
def routed_index_to_indices(index, delta: int = 2, m: int = 2):
    """
    In the routed setting, convert a 1-D index to the corresponding a,b,x,y,z indices
    """
    a = (index // (delta * m**2)) % delta
    b = (index // (m**2)) % delta
    x = (index // m) % m
    y = index % m
    z = index // (delta**2 * m**2)
    return a, b, x, y, z


# Given a set of indices, return the corresponding 1-D index
def routed_indices_to_index(a, b, x, y, z, delta: int = 2, m: int = 2):
    """
    In the routed setting, convert a set of indices to the corresponding 1-D index
    """
    return (z * (delta**2 * m**2)) + (a * (delta * m**2)) + (b * (m**2)) + (x * m) + y


# Certain typical behaviors

# The maximally mixed state over the experiment space
completely_mixed_behavior = Behavior((1 / 4) * np.ones(32))

# The usual (2,2,2) PR box
SR_pr_box = np.array(
    [1 / 2, 1 / 2, 1 / 2, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 0]
)

# The PR box in the experiment space : p(ab|xy) is assumed to be
# independent of the value of z
pr_box = Behavior(np.concatenate((SR_pr_box, SR_pr_box), axis=0))


def display_ns_test_arrays(sum_over_b: bool = False, verbose: bool = False):
    """
    Only serves to show the effects on an array of the operations used in Behavior.no_signaling(),
    with delta=2 and m=2.
    """
    axis_of_sum = 2 if sum_over_b else 1
    axis_to_move = 0 if sum_over_b else 2
    final_shape = (4, 4) if sum_over_b else (8, 2)

    check: np.ndarray = np.array(
        [
            [
                [
                    ["p0000S", "p0001S", "p0010S", "p0011S"],
                    ["p0100S", "p0101S", "p0110S", "p0111S"],
                ],
                [
                    ["p1000S", "p1001S", "p1010S", "p1011S"],
                    ["p1100S", "p1101S", "p1110S", "p1111S"],
                ],
            ],
            [
                [
                    ["p0000L", "p0001L", "p0010L", "p0011L"],
                    ["p0100L", "p0101L", "p0110L", "p0111L"],
                ],
                [
                    ["p1000L", "p1001L", "p1010L", "p1011L"],
                    ["p1100L", "p1101L", "p1110L", "p1111L"],
                ],
            ],
        ],
        dtype=object,
    )

    check = check.reshape(2, 2, 2, 2, 2)

    if verbose:
        print(check.reshape(2, 4, 4))
        print("\n------\n")
        print(check.shape)
        print("\n------\n")
        print(check.sum(axis=axis_of_sum))
        print("\n------\n")
        print("")
        print(np.moveaxis(check.sum(axis=axis_of_sum), axis_to_move, -1))
        print("\n------\n")
    print(np.moveaxis(check.sum(axis=axis_of_sum), axis_to_move, -1).reshape(final_shape))


if __name__ == "__main__":
    check: np.ndarray = np.array(
        [
            [
                [
                    ["p0000S", "p0001S", "p0010S", "p0011S"],
                    ["p0100S", "p0101S", "p0110S", "p0111S"],
                ],
                [
                    ["p1000S", "p1001S", "p1010S", "p1011S"],
                    ["p1100S", "p1101S", "p1110S", "p1111S"],
                ],
            ],
            [
                [
                    ["p0000L", "p0001L", "p0010L", "p0011L"],
                    ["p0100L", "p0101L", "p0110L", "p0111L"],
                ],
                [
                    ["p1000L", "p1001L", "p1010L", "p1011L"],
                    ["p1100L", "p1101L", "p1110L", "p1111L"],
                ],
            ],
        ],
        dtype=object,
    )

    check = check.reshape(2, 4, 4)

    check = Behavior(check)
    print(check)
    print(check.get_vector())
