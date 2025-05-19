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

    def get_vector_element(self, index: int):
        if self.behavior_vector is None:
            raise ValueError("Behavior vector is not initialized.")
        return self.behavior_vector[index]

    @abstractmethod
    def get_matrix_element(self, indices: tuple):
        """
        In the matrix representation, get the element at the given indices.
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

    def get_matrix_element(self, indices: tuple):
        """
        In the matrix representation, get the element at the given indices.
        For RoutedBehavior, the indices are expected in the form (z, a, b, x, y).
        """
        if len(indices) != 5:
            raise ValueError("Expected 5 indices (z, a, b, x, y).")
        a, b, x, y, z = indices
        return self.behavior_vector_to_matrix(self.behavior_vector)[
            z, self.delta * a + b, self.m * x + y
        ]

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
            if other.delta == self.delta and other.m == self.m:
                return other
            else:
                logger.warning(
                    f"RoutedBehavior with different delta or m: ({other.delta}, {other.m}) != ({self.delta}, {self.m})"  # noqa: E501
                )
        elif isinstance(other, np.ndarray):
            if other.shape == self.vector_shape:
                return RoutedBehavior(self.delta, self.m, other)
            elif other.shape == self.matrix_shape:
                return RoutedBehavior(self.delta, self.m, self.behavior_matrix_to_vector(other))
            logger.warning(f"Invalid shape on conversion into Behavior: {other.shape}")
        elif isinstance(other, int) or isinstance(other, float):
            return RoutedBehavior(self.delta, self.m, np.ones(self.vector_shape) * other)

        else:
            logger.warning(
                f"Attempted to convert {other} into Behavior for comparison, but it is not a valid type"  # noqa: E501
            )

        return None

    def normalization(self, atol=1e-10, _debug: bool = False):
        """
        Check if the behavior is normalized
        """
        matrix = self.get_matrix()
        if _debug:
            return np.sum(matrix, axis=1)
        return np.all(abs(np.sum(matrix, axis=1) - 1) < atol)

    def no_signaling(self, atol=1e-10, _debug: bool = False):
        summable = np.reshape(self.behavior_vector, (2, self.delta, self.delta, self.m, self.m))

        sum_over_a: np.ndarray = np.sum(summable, axis=1)
        sum_over_a = np.moveaxis(sum_over_a, 2, -1)
        # Move the x axis to the end
        sum_over_a = sum_over_a.reshape(2 * self.delta * self.m, self.m)
        # Reshape to have every row as a different x

        sum_over_b: np.ndarray = np.sum(summable, axis=2)
        sum_over_b = np.moveaxis(sum_over_b, 0, -1)
        # Move the z axis to the end
        # The y axis is already at the end
        sum_over_b = sum_over_b.reshape(self.delta * self.m, 2 * self.m)
        # Reshape to have every row as a different (y,z)

        if _debug:
            return (
                sum_over_a,
                sum_over_b,
            )

        a_checksum = np.abs(sum_over_a - sum_over_a[:, [0]]) < atol
        b_checksum = np.abs(sum_over_b - sum_over_b[:, [0]]) < atol

        a_no_signaling = np.all(a_checksum, axis=1)
        b_no_signaling = np.all(b_checksum, axis=1)
        return np.all(a_no_signaling) and np.all(b_no_signaling)


class LatentSRNSBehavior(Behavior):
    """
    Class to represent the latent behavior q such that:
    p is SRNS iff p=f(q)
    Assimilated to the vector (q(ab|xy), q(a beta|x))
    """

    def __init__(self, delta: int, m: int, vector: np.ndarray = None):
        """
        Initialize the behavior with delta and m parameters.
        :param delta: The number of possible outputs for Alice and Bob
        :param m: The number of possible inputs for Alice and Bob
        """
        super().__init__(delta, m, vector)

        self.dim_q_s = self.delta**2 * self.m**2
        self.dim_q_L = self.m * self.delta ** (self.m + 1)
        self.dim_q = self.dim_q_s + self.dim_q_L
        self.vector_shape = (self.dim_q,)
        assert (
            self.behavior_vector.shape == (self.dim_q,)
        ), f"Invalid shape {self.behavior_vector.shape}. Expected {self.vector_shape}, to match declared values (delta={self.delta}, m={self.m})."  # noqa: E501

        self.matrix_shapes = [(self.delta**2, self.m**2), (self.delta ** (self.m + 1), self.m)]

    def behavior_vector_to_matrix(self, behavior_vector):
        q_short = behavior_vector[: self.dim_q_s]
        q_long = behavior_vector[self.dim_q_s :]

        q_short = np.reshape(q_short, self.matrix_shapes[0])
        q_long = np.reshape(q_long, self.matrix_shapes[1])
        return q_short, q_long

    def behavior_matrix_to_vector(self, behavior_matrix):
        q_short = behavior_matrix[0]
        q_long = behavior_matrix[1]

        q_short = np.reshape(q_short, self.dim_q_s)
        q_long = np.reshape(q_long, self.dim_q_L)
        return np.concatenate((q_short, q_long), axis=0)

    def get_matrix_element(self, indices: tuple):
        """
        In the matrix representation, get the element at the given indices.
        For LatentSRNSBehavior, the indices are expected in the forms :
        - (z, a, b, x, y) for the short path
        - (z, a, beta, x) for the long path, with beta a tuple-like object
                          with m elements, each in [0, delta-1]
        """
        z = indices[0]
        loc_tuple = indices[1:]
        mat_form = self.behavior_vector_to_matrix(self.behavior_vector)
        if z == 0:
            if len(loc_tuple) != 4:
                raise ValueError("Expected 4 indices (a, b, x, y) to locate on short path.")
            a, b, x, y = loc_tuple
            return mat_form[0][self.delta * a + b, self.m * x + y]
        elif z == 1:
            if len(loc_tuple) != 3:
                raise ValueError("Expected 3 indices (a, beta, x) to locate on long path.")
            a, beta, x = loc_tuple
            if len(beta) != self.m:
                raise ValueError(f"Expected {self.m} indices in beta.")
            id_beta = "".join([str(b) for b in beta])
            id_beta = int(id_beta, self.delta)
            if id_beta >= self.delta**self.m:
                raise ValueError(f"Invalid beta index {id_beta} for delta={self.delta}.")
            return mat_form[1][id_beta + (self.delta**self.m * a), x]

    def __str__(self):
        q_short, q_long = self.get_matrix()
        res: str = "Behavior:\n"
        res += f"Short path (z=S):\n{q_short}\n"
        res += f"Long path (z=L) :\n{q_long}\n"
        res += "-" * 12
        return res

    def compare_array(self, other):
        logger.trace(f"Checking if variable {other} can be treated as a Behavior for comparisons")

        if isinstance(other, LatentSRNSBehavior):
            if other.delta == self.delta and other.m == self.m:
                return other
            else:
                logger.warning(
                    f"LatentSRNSBehavior with different delta or m: ({other.delta}, {other.m}) != ({self.delta}, {self.m})"  # noqa: E501
                )
        elif isinstance(other, np.ndarray):
            if other.shape == self.vector_shape:
                return LatentSRNSBehavior(self.delta, self.m, other)
        elif (
            isinstance(other, list)
            and len(other) == 2
            and isinstance(other[0], np.ndarray)
            and isinstance(other[1], np.ndarray)
        ):
            if other[0].shape == self.matrix_shapes[0] and other[1].shape == self.matrix_shapes[1]:
                return LatentSRNSBehavior(
                    self.delta, self.m, vector=self.behavior_matrix_to_vector(other)
                )

            logger.warning(
                f"A list of arrays was passed, but the shapes are invalid: ({other[0].shape}, {other[1].shape}) != ({self.matrix_shapes[0]}, {self.matrix_shapes[1]})"  # noqa: E501
            )

        elif isinstance(other, int) or isinstance(other, float):
            return LatentSRNSBehavior(self.delta, self.m, np.ones(self.vector_shape) * other)

        else:
            logger.warning(
                f"Attempted to convert {other} into Behavior for comparison, but it is not a valid type (accepted types are int, float, np.ndarray, list of 2 np.ndarrays, and matching LatentSRNSBehavior)"  # noqa: E501
            )

        return None

    def normalization(self, atol=1e-10, _debug: bool = False):
        """
        Check if the behavior is normalized
        """
        q_short, q_long = self.behavior_vector_to_matrix(self.behavior_vector)
        if _debug:
            return np.sum(q_short, axis=1), np.sum(q_long, axis=1)
            # Check the aspect of the sum to verify we check the correct sum
        return np.all(abs(np.sum(q_short, axis=1) - 1) < atol) and np.all(
            abs(np.sum(q_long, axis=1) - 1) < atol
        )

    def no_signaling(self, atol=1e-10, _debug: bool = False):
        # TODO: Implement the no-signaling condition for the latent behavior
        q_short, q_long = self.behavior_vector_to_matrix(self.behavior_vector)

        # No-signaling condition on Alice's side
        # ie : sum_a q(ab|xyz) = q(b|yz)/q(beta|z)
        sum_over_a_short = np.sum(q_short.reshape(self.delta, self.delta, self.m, self.m), axis=0)
        sum_over_a_short = np.moveaxis(sum_over_a_short, 1, -1)
        # Move the x axis to the end
        sum_over_a_short = sum_over_a_short.reshape(self.delta * self.m, self.m)
        # Reshape to have every row as a different x

        sum_over_a_long = np.sum(q_long.reshape(self.delta, self.delta**self.m, self.m), axis=0)
        # Array already has correct shape : one row per x

        # No-signaling condition on Bob's side
        # With the short path, we have:
        sum_over_b_short = np.sum(q_short.reshape(self.delta, self.delta, self.m, self.m), axis=1)
        sum_over_b_short = sum_over_b_short.reshape(self.delta * self.m, self.m)
        # With the long path, we have:
        sum_over_b_long = np.sum(q_long.reshape(self.delta, self.delta**self.m, self.m), axis=1)
        sum_over_b_long = sum_over_b_long.reshape(self.delta * self.m)
        # Now we must check that for every (a,x), q(a|x) is well defined,
        # ie independent of the value of z
        sum_over_b = np.column_stack((sum_over_b_short, sum_over_b_long))

        if _debug:
            return (
                sum_over_a_short,
                sum_over_a_long,
                sum_over_b,
            )

        a_checksum_short = np.abs(sum_over_a_short - sum_over_a_short[:, [0]]) < atol
        a_checksum_long = np.abs(sum_over_a_long - sum_over_a_long[:, [0]]) < atol
        b_checksum = np.abs(sum_over_b - sum_over_b[:, [0]]) < atol

        a_no_signaling_short = np.all(a_checksum_short, axis=1)
        a_no_signaling_long = np.all(a_checksum_long, axis=1)
        b_no_signaling = np.all(b_checksum, axis=1)

        return (
            np.all(a_no_signaling_short) and np.all(a_no_signaling_long) and np.all(b_no_signaling)
        )


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
completely_mixed_behavior = RoutedBehavior(
    delta=2,
    m=2,
    vector=(1 / 4) * np.ones(32),
)

# The usual (2,2,2) PR box
SR_pr_box = np.array(
    [1 / 2, 1 / 2, 1 / 2, 0, 0, 0, 0, 1 / 2, 0, 0, 0, 1 / 2, 1 / 2, 1 / 2, 1 / 2, 0]
)

# The PR box in the experiment space : p(ab|xy) is assumed to be
# independent of the value of z
pr_box = RoutedBehavior(
    delta=2,
    m=2,
    vector=np.concatenate((SR_pr_box, SR_pr_box), axis=0),
)


def display_ns_test_arrays(sum_over_b: bool = False, verbose: bool = False):
    """
    Only serves to show the effects on an array of the operations used in Behavior.no_signaling(),
    with delta=2 and m=2.
    """
    # OBSOLETE WITH THE _DEBUG ARGUMENT ADDED TO NO_SIGNALING CHECKS
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

    check = RoutedBehavior(
        delta=2,
        m=2,
        vector=check.reshape(
            32,
        ),
    )
    # print(check)
    # print(check.get_vector())
    print(*check.no_signaling(_debug=True))

    print("\n\n------\n\n")

    check_latent_short: np.ndarray = np.array(
        [
            ["p0000S", "p0001S", "p0010S", "p0011S"],
            ["p0100S", "p0101S", "p0110S", "p0111S"],
            ["p1000S", "p1001S", "p1010S", "p1011S"],
            ["p1100S", "p1101S", "p1110S", "p1111S"],
        ]
    )
    check_latent_long: np.ndarray = np.array(
        [
            ["p0(00)0L", "p0(00)1L"],
            ["p0(01)0L", "p0(01)1L"],
            ["p0(10)0L", "p0(10)1L"],
            ["p0(11)0L", "p0(11)1L"],
            ["p1(00)0L", "p1(00)1L"],
            ["p1(01)0L", "p1(01)1L"],
            ["p1(10)0L", "p1(10)1L"],
            ["p1(11)0L", "p1(11)1L"],
        ],
        dtype=object,
    )

    check_latent_vect: np.ndarray = np.concatenate(
        (
            check_latent_short.reshape(16),
            check_latent_long.reshape(16),
        ),
        axis=0,
    )
    check_latent: LatentSRNSBehavior = LatentSRNSBehavior(
        delta=2,
        m=2,
        vector=check_latent_vect,
    )
    # print(check_latent)
    # print(check_latent.get_vector())
    # print(check_latent.get_matrix_element((1, 1, (1, 0), 1)))
    sum_over_a_short, sum_over_a_long, sum_over_b = check_latent.no_signaling(_debug=True)
    print(sum_over_a_short)
    print()
    print(sum_over_a_long)
    print()
    print(sum_over_b)
