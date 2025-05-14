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
    def behavior_vector_to_matrix(behavior_vector):
        """
        Convert a vector behavior to its matrix representation
        :param behavior_vector: The behavior vector
        :return: The behavior matrix
        """
        return np.reshape(behavior_vector, (2, 4, 4))

    # ... and vice versa
    def behavior_matrix_to_vector(behavior_matrix):
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
        return np.all(np.sum(matrix, axis=1) == 1)

    def no_signaling(self):
        """
        Check if the behavior is no-signaling
        """
        all_indices = self.get_all_indices()
        # No-signaling condition from Alice to Bob
        # TODO


# The maximally mixed state over the experiment space
I = (1 / 4) * np.ones(32)  # noqa: E741

# The usual (2,2,2) PR box
SR_pr_box = np.array(
    [
        1 / 2,
        1 / 2,
        1 / 2,
        0,
        0,
        0,
        0,
        1 / 2,
        0,
        0,
        0,
        1 / 2,
        1 / 2,
        1 / 2,
        1 / 2,
        0,
    ]
)

# The PR box in the experiment space : p(ab|xy) is assumed to be
# independent of the value of z
pr_box = np.concatenate((SR_pr_box, SR_pr_box), axis=0)
