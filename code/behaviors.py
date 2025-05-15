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
        return np.reshape(behavior_vector, (2, 4, 4))

    # ... and vice versa
    def behavior_matrix_to_vector(self, behavior_matrix):
        """
        Convert a matrix behavior to its vector representation
        :param behavior_matrix: The behavior matrix
        :return: The behavior vector
        """
        return np.reshape(behavior_matrix, (32,))

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
        return np.array([self.index_to_indices(i) for i in range(32)])

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

    # COORDINATE UTILS

    # Given a 1-D index, return the corresponding a,b,x,y,z indices
    def index_to_indices(self, index):
        """
        Given a 1-D index, return the corresponding a,b,x,y,z indices
        :param index: The 1-D index
        :return: The corresponding a,b,x,y,z indices
        """
        # WARNING : This function works only for delta=m=2
        a = (index // 8) % 2
        b = (index // 4) % 2
        x = (index // 2) % 2
        y = index % 2
        z = index // 16
        return a, b, x, y, z

    # Given a set of indices, return the corresponding 1-D index
    def indices_to_index(self, a, b, x, y, z):
        """
        Given a set of indices, return the corresponding 1-D index
        :return: The corresponding 1-D index
        """
        # WARNING : This function works only for delta=m=2
        return (z * 16) + (a * 8) + (b * 4) + (x * 2) + y

    # CONDITIONS
    def positivity(self):
        return self >= 0

    def normalization(self):
        """
        Check if the behavior is normalized
        """
        matrix = self.get_matrix()
        return np.all(abs(np.sum(matrix, axis=1) - 1) < 1e-12)

    def no_signaling(self):
        """
        Check if the behavior is no-signaling
        """
        summable = np.reshape(self.behavior_vector, (2, self.delta, self.delta, self.m, self.m))

        logger.debug(f"Summable array shape: {summable.shape}")
        logger.debug(f"Summable array: {summable}")
        sum_over_a: np.ndarray = np.sum(summable, axis=1)
        logger.debug(f"Shape of summed array: {sum_over_a.shape}")
        logger.debug(f"Summed array: {sum_over_a}")
        sum_over_a = np.transpose(sum_over_a, (1, 3, 0, 2))
        logger.debug(f"Transposed summed array shape: {sum_over_a.shape}")
        logger.debug(f"Transposed summed array: {sum_over_a}")
        sum_over_a = sum_over_a.reshape(-1, self.m)
        logger.debug(f"Reshaped summed array shape: {sum_over_a.shape}")
        logger.debug(f"Reshaped summed array: {sum_over_a}")
        a_no_signaling = np.all(sum_over_a == sum_over_a[:, [0]], axis=1)

        sum_over_b: np.ndarray = np.sum(summable, axis=2)
        sum_over_b = np.transpose(sum_over_b, (1, 3, 0, 2))
        sum_over_b = sum_over_b.reshape(-1, self.m)
        b_no_signaling = np.all(sum_over_b == sum_over_b[:, [0]], axis=1)

        return np.all(a_no_signaling) and np.all(b_no_signaling)

    def is_normalized(self):
        return self.positivity() and self.normalization()

    def is_no_signaling(self):
        return self.positivity() and self.normalization() and self.no_signaling()


def display_ns_test_arrays():
    """
    Only serves to show the effects on an array of the operations used in Behavior.no_signaling()
    """
    check = np.array(
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

    print(check.shape)
    print(check)
    print("\n------\n")
    print(check.reshape(2, 4, 4))
    print("\n------\n")
    print(check.sum(axis=1))
    print("\n------\n")
    print((check.sum(axis=1)).transpose(1, 3, 0, 2))
    print("\n------\n")
    print((check.sum(axis=1)).transpose(1, 3, 0, 2).reshape(-1, 2))
