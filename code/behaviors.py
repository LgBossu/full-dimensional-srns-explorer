import numpy as np
from loguru import logger


class Behavior:
    """
    Class to represent a behavior in the experiment space
    """

    # BASIC CLASS PARAMETERS

    m: int = 2  # The number of possible inputs for Alice and Bob
    delta: int = 2  # The number of possible outputs for Alice and Bob

    vector_shape: tuple[int] = (2 * delta**2 * m**2,)  # The dimension of the experiment space
    matrix_shape: tuple[int, int, int] = (2, delta**2, m**2)  # The shape of the behavior matrix

    # VECTOR AND MATRIX FORMS CONVERSIONS

    # Convert a vector behavior to its matrix representation
    def behavior_vector_to_matrix(self, behavior_vector):
        """
        Convert a vector behavior to its matrix representation
        :param behavior_vector: The behavior vector
        :return: The behavior matrix
        """
        return np.reshape(behavior_vector, (2, self.delta**2, self.m**2))

    # ... and vice versa
    def behavior_matrix_to_vector(self, behavior_matrix):
        """
        Convert a matrix behavior to its vector representation
        :param behavior_matrix: The behavior matrix
        :return: The behavior vector
        """
        return np.reshape(behavior_matrix, (2 * self.delta**2 * self.m**2,))

    # CONSTRUCTOR

    def __init__(self, coords_array: np.ndarray):
        """
        Initialize the behavior with a vector or matrix representation
        :param coords_array: The behavior vector or matrix
        """
        if coords_array.shape == self.vector_shape:
            self.behavior_vector = coords_array
        elif coords_array.shape == self.matrix_shape:
            self.behavior_vector = self.behavior_matrix_to_vector(coords_array)
        else:
            raise ValueError(
                f"Invalid shape {coords_array.shape}. Expected {self.vector_shape} or {self.matrix_shape}"  # noqa: E501
            )

    # GETTERS
    def get_vector(self):
        """
        Get the behavior vector
        :return: The behavior vector
        """
        return self.behavior_vector

    def get_matrix(self):
        """
        Get the behavior matrix
        :return: The behavior matrix
        """
        return self.behavior_vector_to_matrix(self.behavior_vector)

    def get_all_indices(self):
        return np.array([self.index_to_indices(i) for i in range(2 * self.delta**2 * self.m**2)])

    # STRING REPRESENTATIONS

    def __repr__(self):
        return f"Behavior({self.behavior_vector})"

    def __str__(self):
        matrix_form = self.behavior_vector_to_matrix(self.behavior_vector)
        res: str = "Behavior:\n"
        res += f"Short path (z=S):\n{matrix_form[0]}\n"
        res += f"Long path (z=L) :\n{matrix_form[1]}\n"
        res += "-" * 12
        return res

    # EQUALITY
    def compare_array(self, other):
        logger.debug(f"Checking if variable {other} can be treated as a Behavior for comparisons")

        if isinstance(other, Behavior):
            return other
        elif isinstance(other, np.ndarray):
            if other.shape == self.vector_shape or other.shape == self.matrix_shape:
                return Behavior(other)
            logger.warning(f"Invalid shape on conversion into Behavior: {other.shape}")
        elif isinstance(other, int) or isinstance(other, float):
            return Behavior(np.ones(self.vector_shape) * other)

        logger.warning(
            f"Attempted to convert {other} into Behavior for comparison, but it is not a valid type"
        )
        return None

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

    def normalization(self):
        """
        Check if the behavior is normalized
        """
        matrix = self.get_matrix()
        return np.all(abs(np.sum(matrix, axis=1) - 1) < 1e-12)

    def no_signaling(self, atol=1e-8):
        """
        Check if the behavior is no-signaling
        """
        summable = np.reshape(self.behavior_vector, (2, self.delta, self.delta, self.m, self.m))

        logger.debug(f"Summable array shape: {summable.shape}")
        logger.debug(f"Summable array: {summable}")
        sum_over_a: np.ndarray = np.sum(summable, axis=1)
        logger.debug(f"Shape of summed array: {sum_over_a.shape}")
        logger.debug(f"Summed array: {sum_over_a}")
        sum_over_a = np.moveaxis(sum_over_a, 2, -1)
        # logger.debug(f"Transposed summed array shape: {sum_over_a.shape}")
        logger.debug(f"Reordered summed array: {sum_over_a}")
        sum_over_a = sum_over_a.reshape(2 * self.delta * self.m, self.m)
        logger.debug(f"Reshaped summed array shape: {sum_over_a.shape}")
        logger.debug(f"Reshaped summed array: {sum_over_a}")
        a_checksum = np.abs(sum_over_a - sum_over_a[:, [0]]) < atol
        a_no_signaling = np.all(a_checksum, axis=1)

        sum_over_b: np.ndarray = np.sum(summable, axis=2)
        sum_over_b = np.moveaxis(sum_over_b, 0, -1)
        sum_over_b = sum_over_b.reshape(self.delta * self.m, 2 * self.m)
        b_checksum = np.abs(sum_over_b - sum_over_b[:, [0]]) < atol
        b_no_signaling = np.all(b_checksum, axis=1)

        return np.all(a_no_signaling) and np.all(b_no_signaling)

    def is_normalized(self):
        return self.positivity() and self.normalization()

    def is_no_signaling(self, atol=1e-8):
        return self.positivity() and self.normalization() and self.no_signaling(atol=atol)


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
