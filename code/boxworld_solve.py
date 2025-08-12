"""
We try to get the facet equations of the short range boxworld set,
to get more insight on how GPTs behave.

The thing is, boxworld's vertices are known to be extremal behaviors with extremal measurements.
An unelegant but effective strategy to solve the SRBoxworld polytope is to enumerate :
- for all vertices
- for all measurement strategies (Alice, Bob short, Bob long channel)
- for all [deterministic] transformations along Bob long channel

the p vectors of probabilities. These live in the observable experiment space,
so once we have them, we can just (deduplicate and) plug them into the CDD polytope solver.
This is a very brute force approach, but it should work, assuming it is sufficient to consider only:
- deterministic degradations
- deterministic transformations
along the Bob long channel.


Reference work on Boxworld description:
```Generalizations of Boxworld
Peter Janotta
Fakultät für Physik und Astronomie
Universität Würzburg
Am Hubland
97074 Würzburg
Germany```
"""

# flake8: noqa: E501

import itertools
from enum import Enum
from typing import Union

# from behaviors import routed_indices_to_index, short_range_indices_to_index
import numpy as np
from loguru import logger
from tqdm import tqdm


class ArrayWrapper:
    """Though a bit heavy, this wrapper simply helps us to use numpy arrays as enum values,
    while still allowing for equality checks and hashing."""

    def __init__(self, arr):
        self.arr = np.array(arr)

    def __eq__(self, other):
        if not isinstance(other, ArrayWrapper):
            return False
        return np.array_equal(self.arr, other.arr)

    def __hash__(self):
        return hash(self.arr.tobytes())


class BoxworldVertex(Enum):
    """
    The vertices of the regular boxworld polytope.
    """

    # As per `Generalizations of Boxworld`
    w1 = ArrayWrapper(np.array([1, 0, 1]))
    w2 = ArrayWrapper(np.array([0, 1, 1]))
    w3 = ArrayWrapper(np.array([-1, 0, 1]))
    w4 = ArrayWrapper(np.array([0, -1, 1]))
    # # As per `General Probabilistic Theories - An Introduction`
    # w1 = ArrayWrapper(np.array([0, 0, 1]))
    # w2 = ArrayWrapper(np.array([1, 0, 1]))
    # w3 = ArrayWrapper(np.array([0, 1, 1]))
    # w4 = ArrayWrapper(np.array([1, 1, 1]))


class BoxworldEffect(Enum):
    """
    The effects of the boxworld polytope.
    """

    # As per `Generalizations of Boxworld`
    u = ArrayWrapper(np.array([0, 0, 1]))
    zero = ArrayWrapper(
        np.array([0, 0, 0])
    )  # For normalization, we need zero as u's complement effect
    e1 = ArrayWrapper((1 / 2) * np.array([-1, -1, 1]))
    e2 = ArrayWrapper((1 / 2) * np.array([1, -1, 1]))
    e3 = ArrayWrapper((1 / 2) * np.array([1, 1, 1]))
    e4 = ArrayWrapper((1 / 2) * np.array([-1, 1, 1]))
    # # As per `General Probabilistic Theories - An Introduction`
    # u = ArrayWrapper(np.array([0, 0, 1]))
    # zero = ArrayWrapper(np.array([0, 0, 0]))
    # e1 = ArrayWrapper(np.array([1, 0, 0]))
    # e2 = ArrayWrapper(np.array([0, 1, 0]))
    # e3 = ArrayWrapper(np.array([-1, 0, 1]))
    # e4 = ArrayWrapper(np.array([0, -1, 1]))


# class ComputedEffect:
#     def __init__(self, value: np.ndarray):
#         self.value = ArrayWrapper(value)

#     def __eq__(self, other: object) -> bool:
#         if not isinstance(other, Union[BoxworldEffect, "ComputedEffect"]):
#             return False
#         return self.value == other.value


# Effect = Union[BoxworldEffect, ComputedEffect]
# BoxworldObject = Union[BoxworldVertex, Effect]
BoxworldObject = Union[BoxworldVertex, BoxworldEffect]


def tensor_product(a: BoxworldObject, b: BoxworldObject) -> np.ndarray:
    """
    Compute the tensor product of two boxworld vertices.
    This is used to build the product space from the boxworld polytope.

    As of now, we use the maximal tensor product on vertices (so all points compatible
    with local measurements).
    This implies the product over effects is minimal (ie, only separable measurements are allowed).
    """
    a_arr = a.value.arr
    b_arr = b.value.arr
    return np.kron(a_arr, b_arr)


class BoxworldBipartiteVertex(Enum):
    """
    The bipartite boxworld vertices.
    These are the vertices of the bipartite boxworld polytope.
    """

    # First come the sixteen extremal product states
    w1 = ArrayWrapper(tensor_product(BoxworldVertex.w1, BoxworldVertex.w1))  # (1, 1)
    w2 = ArrayWrapper(tensor_product(BoxworldVertex.w1, BoxworldVertex.w2))  # (1, 2)
    w3 = ArrayWrapper(tensor_product(BoxworldVertex.w1, BoxworldVertex.w3))  # (1, 3)
    w4 = ArrayWrapper(tensor_product(BoxworldVertex.w1, BoxworldVertex.w4))  # (1, 4)
    w5 = ArrayWrapper(tensor_product(BoxworldVertex.w2, BoxworldVertex.w1))  # (2, 1)
    w6 = ArrayWrapper(tensor_product(BoxworldVertex.w2, BoxworldVertex.w2))  # (2, 2)
    w7 = ArrayWrapper(tensor_product(BoxworldVertex.w2, BoxworldVertex.w3))  # (2, 3)
    w8 = ArrayWrapper(tensor_product(BoxworldVertex.w2, BoxworldVertex.w4))  # (2, 4)
    w9 = ArrayWrapper(tensor_product(BoxworldVertex.w3, BoxworldVertex.w1))  # (3, 1)
    w10 = ArrayWrapper(tensor_product(BoxworldVertex.w3, BoxworldVertex.w2))  # (3, 2)
    w11 = ArrayWrapper(tensor_product(BoxworldVertex.w3, BoxworldVertex.w3))  # (3, 3)
    w12 = ArrayWrapper(tensor_product(BoxworldVertex.w3, BoxworldVertex.w4))  # (3, 4)
    w13 = ArrayWrapper(tensor_product(BoxworldVertex.w4, BoxworldVertex.w1))  # (4, 1)
    w14 = ArrayWrapper(tensor_product(BoxworldVertex.w4, BoxworldVertex.w2))  # (4, 2)
    w15 = ArrayWrapper(tensor_product(BoxworldVertex.w4, BoxworldVertex.w3))  # (4, 3)
    w16 = ArrayWrapper(tensor_product(BoxworldVertex.w4, BoxworldVertex.w4))  # (4, 4)

    # Then come the eight additional entangled pure states

    # Entangled formulas as per `Generalizations of Boxworld`
    w17 = ArrayWrapper((1 / 2) * (w2.arr - w6.arr + w7.arr + w9.arr))
    w18 = ArrayWrapper((1 / 2) * (w6.arr - w11.arr + w12.arr + w15.arr))
    w19 = ArrayWrapper((1 / 2) * (w1.arr - w6.arr + w7.arr + w10.arr))
    w20 = ArrayWrapper((1 / 2) * (w6.arr - w10.arr + w11.arr + w13.arr))
    w21 = ArrayWrapper((1 / 2) * (w4.arr - w1.arr + w5.arr + w14.arr))
    w22 = ArrayWrapper((1 / 2) * (w4.arr - w1.arr + w6.arr + w13.arr))
    w23 = ArrayWrapper((1 / 2) * (w1.arr - w2.arr + w6.arr + w15.arr))
    w24 = ArrayWrapper((1 / 2) * (w1.arr - w4.arr + w8.arr + w15.arr))


# class BoxworldBipartiteEffect(Enum):
#     """
#     The effects of the bipartite boxworld polytope.
#     These are the effects of the bipartite boxworld polytope.
#     """

#     # Products with the unit effect
#     u = tensor_product(BoxworldEffect.u, BoxworldEffect.u)
#     ue1 = tensor_product(BoxworldEffect.u, BoxworldEffect.e1)
#     ue2 = tensor_product(BoxworldEffect.u, BoxworldEffect.e2)
#     ue3 = tensor_product(BoxworldEffect.u, BoxworldEffect.e3)
#     ue4 = tensor_product(BoxworldEffect.u, BoxworldEffect.e4)
#     e1u = tensor_product(BoxworldEffect.e1, BoxworldEffect.u)
#     e2u = tensor_product(BoxworldEffect.e2, BoxworldEffect.u)
#     e3u = tensor_product(BoxworldEffect.e3, BoxworldEffect.u)
#     e4u = tensor_product(BoxworldEffect.e4, BoxworldEffect.u)

#     # Products with effects
#     e11 = tensor_product(BoxworldEffect.e1, BoxworldEffect.e1)
#     e12 = tensor_product(BoxworldEffect.e1, BoxworldEffect.e2)
#     e13 = tensor_product(BoxworldEffect.e1, BoxworldEffect.e3)
#     e14 = tensor_product(BoxworldEffect.e1, BoxworldEffect.e4)
#     e21 = tensor_product(BoxworldEffect.e2, BoxworldEffect.e1)
#     e22 = tensor_product(BoxworldEffect.e2, BoxworldEffect.e2)
#     e23 = tensor_product(BoxworldEffect.e2, BoxworldEffect.e3)
#     e24 = tensor_product(BoxworldEffect.e2, BoxworldEffect.e4)
#     e31 = tensor_product(BoxworldEffect.e3, BoxworldEffect.e1)
#     e32 = tensor_product(BoxworldEffect.e3, BoxworldEffect.e2)
#     e33 = tensor_product(BoxworldEffect.e3, BoxworldEffect.e3)
#     e34 = tensor_product(BoxworldEffect.e3, BoxworldEffect.e4)
#     e41 = tensor_product(BoxworldEffect.e4, BoxworldEffect.e1)
#     e42 = tensor_product(BoxworldEffect.e4, BoxworldEffect.e2)
#     e43 = tensor_product(BoxworldEffect.e4, BoxworldEffect.e3)
#     e44 = tensor_product(BoxworldEffect.e4, BoxworldEffect.e4)


class TransformOutput(Enum):
    """
    Enumerate all 16 functions mapping two binary inputs (0,1) to a binary output (0,1).
    Each function can be represented by its truth table: (b', y) -> f(b', y),
    where b', y ∈ {0, 1}. There are 4 input pairs, so 2^4 = 16 possible functions.
    """

    f1 = ArrayWrapper(np.array([[0, 0], [0, 0]]))  # Always outputs 0
    f2 = ArrayWrapper(np.array([[0, 0], [0, 1]]))  # Outputs b' AND y
    f3 = ArrayWrapper(np.array([[0, 0], [1, 0]]))  # Outputs NOTy AND b'
    f4 = ArrayWrapper(np.array([[0, 0], [1, 1]]))  # Outputs b'
    f5 = ArrayWrapper(np.array([[0, 1], [0, 0]]))  # Outputs y AND NOTb'
    f6 = ArrayWrapper(np.array([[0, 1], [0, 1]]))  # Outputs y
    f7 = ArrayWrapper(np.array([[0, 1], [1, 0]]))  # Outputs b' XOR y
    f8 = ArrayWrapper(np.array([[0, 1], [1, 1]]))  # Outputs b' OR y
    f9 = ArrayWrapper(np.array([[1, 0], [0, 0]]))  # Outputs NOTy AND NOTb'
    f10 = ArrayWrapper(np.array([[1, 0], [0, 1]]))  # Outputs NOT(b' XOR y)
    f11 = ArrayWrapper(np.array([[1, 0], [1, 0]]))  # Outputs NOTy
    f12 = ArrayWrapper(np.array([[1, 0], [1, 1]]))  # ...
    f13 = ArrayWrapper(np.array([[1, 1], [0, 0]]))
    f14 = ArrayWrapper(np.array([[1, 1], [0, 1]]))
    f15 = ArrayWrapper(np.array([[1, 1], [1, 0]]))
    f16 = ArrayWrapper(np.array([[1, 1], [1, 1]]))


class TransformOutput3(Enum):
    """
    Enumerate all functions mapping tuples (b', y) where b' ∈ {0, 1} and y ∈ {0, 1, 2}
    to a binary output (0, 1).
    Each function can be represented by its truth table: (b', y) -> f(b', y),
    where b' ∈ {0, 1} and y ∈ {0, 1, 2}. There are 6 input pairs, so 2^6 = 64 possible functions.
    """

    f1 = ArrayWrapper(np.array([[0, 0, 0], [0, 0, 0]]))  # Always outputs 0
    f2 = ArrayWrapper(np.array([[0, 0, 0], [0, 0, 1]]))
    f3 = ArrayWrapper(np.array([[0, 0, 0], [0, 1, 0]]))
    f4 = ArrayWrapper(np.array([[0, 0, 0], [0, 1, 1]]))
    f5 = ArrayWrapper(np.array([[0, 0, 0], [1, 0, 0]]))
    f6 = ArrayWrapper(np.array([[0, 0, 0], [1, 0, 1]]))
    f7 = ArrayWrapper(np.array([[0, 0, 0], [1, 1, 0]]))
    f8 = ArrayWrapper(np.array([[0, 0, 0], [1, 1, 1]]))
    f9 = ArrayWrapper(np.array([[0, 0, 1], [0, 0, 0]]))
    f10 = ArrayWrapper(np.array([[0, 0, 1], [0, 0, 1]]))
    f11 = ArrayWrapper(np.array([[0, 0, 1], [0, 1, 0]]))
    f12 = ArrayWrapper(np.array([[0, 0, 1], [0, 1, 1]]))
    f13 = ArrayWrapper(np.array([[0, 0, 1], [1, 0, 0]]))
    f14 = ArrayWrapper(np.array([[0, 0, 1], [1, 0, 1]]))
    f15 = ArrayWrapper(np.array([[0, 0, 1], [1, 1, 0]]))
    f16 = ArrayWrapper(np.array([[0, 0, 1], [1, 1, 1]]))
    f17 = ArrayWrapper(np.array([[0, 1, 0], [0, 0, 0]]))
    f18 = ArrayWrapper(np.array([[0, 1, 0], [0, 0, 1]]))
    f19 = ArrayWrapper(np.array([[0, 1, 0], [0, 1, 0]]))
    f20 = ArrayWrapper(np.array([[0, 1, 0], [0, 1, 1]]))
    f21 = ArrayWrapper(np.array([[0, 1, 0], [1, 0, 0]]))
    f22 = ArrayWrapper(np.array([[0, 1, 0], [1, 0, 1]]))
    f23 = ArrayWrapper(np.array([[0, 1, 0], [1, 1, 0]]))
    f24 = ArrayWrapper(np.array([[0, 1, 0], [1, 1, 1]]))
    f25 = ArrayWrapper(np.array([[0, 1, 1], [0, 0, 0]]))
    f26 = ArrayWrapper(np.array([[0, 1, 1], [0, 0, 1]]))
    f27 = ArrayWrapper(np.array([[0, 1, 1], [0, 1, 0]]))
    f28 = ArrayWrapper(np.array([[0, 1, 1], [0, 1, 1]]))
    f29 = ArrayWrapper(np.array([[0, 1, 1], [1, 0, 0]]))
    f30 = ArrayWrapper(np.array([[0, 1, 1], [1, 0, 1]]))
    f31 = ArrayWrapper(np.array([[0, 1, 1], [1, 1, 0]]))
    f32 = ArrayWrapper(np.array([[0, 1, 1], [1, 1, 1]]))
    f33 = ArrayWrapper(np.array([[1, 0, 0], [0, 0, 0]]))
    f34 = ArrayWrapper(np.array([[1, 0, 0], [0, 0, 1]]))
    f35 = ArrayWrapper(np.array([[1, 0, 0], [0, 1, 0]]))
    f36 = ArrayWrapper(np.array([[1, 0, 0], [0, 1, 1]]))
    f37 = ArrayWrapper(np.array([[1, 0, 0], [1, 0, 0]]))
    f38 = ArrayWrapper(np.array([[1, 0, 0], [1, 0, 1]]))
    f39 = ArrayWrapper(np.array([[1, 0, 0], [1, 1, 0]]))
    f40 = ArrayWrapper(np.array([[1, 0, 0], [1, 1, 1]]))
    f41 = ArrayWrapper(np.array([[1, 0, 1], [0, 0, 0]]))
    f42 = ArrayWrapper(np.array([[1, 0, 1], [0, 0, 1]]))
    f43 = ArrayWrapper(np.array([[1, 0, 1], [0, 1, 0]]))
    f44 = ArrayWrapper(np.array([[1, 0, 1], [0, 1, 1]]))
    f45 = ArrayWrapper(np.array([[1, 0, 1], [1, 0, 0]]))
    f46 = ArrayWrapper(np.array([[1, 0, 1], [1, 0, 1]]))
    f47 = ArrayWrapper(np.array([[1, 0, 1], [1, 1, 0]]))
    f48 = ArrayWrapper(np.array([[1, 0, 1], [1, 1, 1]]))
    f49 = ArrayWrapper(np.array([[1, 1, 0], [0, 0, 0]]))
    f50 = ArrayWrapper(np.array([[1, 1, 0], [0, 0, 1]]))
    f51 = ArrayWrapper(np.array([[1, 1, 0], [0, 1, 0]]))
    f52 = ArrayWrapper(np.array([[1, 1, 0], [0, 1, 1]]))
    f53 = ArrayWrapper(np.array([[1, 1, 0], [1, 0, 0]]))
    f54 = ArrayWrapper(np.array([[1, 1, 0], [1, 0, 1]]))
    f55 = ArrayWrapper(np.array([[1, 1, 0], [1, 1, 0]]))
    f56 = ArrayWrapper(np.array([[1, 1, 0], [1, 1, 1]]))
    f57 = ArrayWrapper(np.array([[1, 1, 1], [0, 0, 0]]))
    f58 = ArrayWrapper(np.array([[1, 1, 1], [0, 0, 1]]))
    f59 = ArrayWrapper(np.array([[1, 1, 1], [0, 1, 0]]))
    f60 = ArrayWrapper(np.array([[1, 1, 1], [0, 1, 1]]))
    f61 = ArrayWrapper(np.array([[1, 1, 1], [1, 0, 0]]))
    f62 = ArrayWrapper(np.array([[1, 1, 1], [1, 0, 1]]))
    f63 = ArrayWrapper(np.array([[1, 1, 1], [1, 1, 0]]))
    f64 = ArrayWrapper(np.array([[1, 1, 1], [1, 1, 1]]))


complements = {
    BoxworldEffect.e1: BoxworldEffect.e3,
    BoxworldEffect.e2: BoxworldEffect.e4,
    BoxworldEffect.e3: BoxworldEffect.e1,
    BoxworldEffect.e4: BoxworldEffect.e2,
    BoxworldEffect.u: BoxworldEffect.zero,
    BoxworldEffect.zero: BoxworldEffect.u,
}


def generate_all_extremal_points(
    delta: int = 2,
    m_alice: int = 2,
    m_bob_short: int = 2,
    m_bob_long: int = 2,
    idx_range: tuple[int, int] = (0, 23),
) -> list[np.ndarray]:
    """
    Generate all extremal points of the routed Bell experiment boxworld strategies.
    Parameters prefixed `m` are the numbers of inputs for respectively Alice, Bob short and Bob long channels.
    """

    def generate_distribution_symmetries_and_concatenate(
        distrib_short: np.ndarray, distrib_long: np.ndarray
    ) -> list[np.ndarray]:
        """
        Generate all relabelings (symmetries) of a distribution by permuting Alice/Bob inputs x and y.
        Returns all unique permutations as flattened arrays.
        Long-path distributions ARE required to be symmetrized along, to account for Alice-sided symmetries.

        :param distribution: The SHORT-PATH distribution to generate symmetries for.
        :param distrib_long: The LONG-PATH distribution to generate symmetries for.
        :return: A list of all symmetrized concatenated distributions.
        """
        shaped_short = distrib_short.reshape((delta, delta, m_alice, m_bob_short))
        shaped_long = distrib_long.reshape((delta, delta, m_alice, m_bob_long))
        symmetries: list[np.ndarray] = []
        x_perms = list(itertools.permutations(range(m_alice)))
        y_perms = list(itertools.permutations(range(m_bob_short)))
        for x_perm in x_perms:
            permuted_long = shaped_long[:, :, x_perm, :]
            for y_perm in y_perms:
                # Permute x and y axes
                permuted_short = shaped_short[:, :, x_perm, :][:, :, :, y_perm]
                symmetries.append(np.hstack((permuted_short.flatten(), permuted_long.flatten())))

        # Optionally, remove duplicates
        unique = []
        seen = set()
        for arr in symmetries:
            key = arr.tobytes()
            if key not in seen:
                unique.append(arr)
                seen.add(key)
        return unique

    def routed_indices_to_index(a, b, x, y, z):
        """
        In the routed setting, convert a set of indices to the corresponding 1-D index
        """
        if z == 0:
            return (
                (a * (delta * m_alice * m_bob_short))
                + (b * (m_alice * m_bob_short))
                + (x * m_alice)
                + y
            )
        elif z == 1:
            return (
                (a * (delta * m_alice * m_bob_long))
                + (b * (m_alice * m_bob_long))
                + (x * m_alice)
                + y
            )

    computed_distributions = []  # This will hold all computed effects

    vertices = list(BoxworldBipartiteVertex)[idx_range[0] : idx_range[1] + 1]

    for state in vertices:
        logger.info(f"Processing state: {state.value.arr}")
        # Only iterate over canonical measurement tuples (no redundant permutations)
        from itertools import combinations_with_replacement

        canonical_measurements_alice = list(combinations_with_replacement(BoxworldEffect, m_alice))
        canonical_measurements_bob = list(
            combinations_with_replacement(BoxworldEffect, m_bob_short)
        )

        for alice_tuple, bob_short_tuple in tqdm(
            [(i, j) for i in canonical_measurements_alice for j in canonical_measurements_bob]
        ):
            alice = [
                [alice_tuple[i].value.arr, complements[alice_tuple[i]].value.arr]
                for i in range(m_alice)
            ]
            bob_short = [
                [bob_short_tuple[i].value.arr, complements[bob_short_tuple[i]].value.arr]
                for i in range(m_bob_short)
            ]

            for bob_long_measurement in BoxworldEffect:
                bob_long = [
                    bob_long_measurement.value.arr,
                    complements[bob_long_measurement].value.arr,
                ]

                if m_bob_long not in [2, 3]:
                    raise ValueError("Only m=2 and m=3 are supported for now.")
                TranformClass = {2: TransformOutput, 3: TransformOutput3}[m_bob_long]

                for transform in TranformClass:  # We pick bob's deterministic transformation from the degraded state to the final output
                    # Compute the output probabilities for the current configuration
                    distrib_short = np.zeros(delta**2 * m_alice * m_bob_short, dtype=float)
                    distrib_long = np.zeros(delta**2 * m_alice * m_bob_long, dtype=float)

                    # Compute the short-path distribution
                    # logger.debug("Computing short-path distribution")
                    for a, b, x, y in [
                        (i, j, k, l)
                        for i in range(delta)
                        for j in range(delta)
                        for k in range(m_alice)
                        for l in range(m_bob_short)  # noqa: E741
                    ]:
                        distrib_short[
                            routed_indices_to_index(
                                a,
                                b,
                                x,
                                y,
                                0,  # 0 for short-path
                            )
                        ] = np.dot(np.kron(alice[x][a], bob_short[y][b]), state.value.arr)

                    # Compute the long-path distribution
                    # logger.debug("Computing long-path distribution")
                    for a, b, x, y in [
                        (i, j, k, l)
                        for i in range(delta)
                        for j in range(delta)
                        for k in range(m_alice)
                        for l in range(m_bob_long)  # noqa: E741
                    ]:
                        for b_prime in range(delta):
                            distrib_long[
                                routed_indices_to_index(
                                    a,
                                    transform.value.arr[b_prime, y],
                                    x,
                                    y,
                                    1,  # 1 for long-path
                                )
                            ] += np.dot(
                                np.kron(alice[x][a], bob_long[b_prime]), state.value.arr
                            )  # we divide by two because somehow long-path final results sum to two and usually require renormalization, i'm not quite sure why # TODO : check the logic of long-path distribution computation

                    # Check normalization on the short-path
                    # it should already be normalized
                    normalized = True
                    for x, y in [(i, j) for i in range(m_alice) for j in range(m_bob_short)]:
                        total = np.sum(
                            [
                                distrib_short[i]
                                for i in [
                                    routed_indices_to_index(_a, _b, x, y, 0)
                                    for _a in range(delta)
                                    for _b in range(delta)
                                ]
                            ]
                        )
                        if total != 1:
                            logger.error(f"Distribution not normalized: {total} != 1")
                            normalized = False

                    # Normalize the long-path
                    for x, y in [(i, j) for i in range(m_alice) for j in range(m_bob_long)]:
                        total = np.sum(
                            [
                                distrib_long[i]
                                for i in [
                                    routed_indices_to_index(_a, _b, x, y, 1)
                                    for _a in range(delta)
                                    for _b in range(delta)
                                ]
                            ]
                        )
                        if total > 0:
                            if total != 1:
                                logger.trace(f"Normalizing long-path distribution: {total} != 1")
                                # Normalize the long-path distribution
                                for i in [
                                    routed_indices_to_index(_a, _b, x, y, 1)
                                    for _a in range(delta)
                                    for _b in range(delta)
                                ]:
                                    distrib_long[i] /= total
                        else:
                            logger.error(f"Long-path distribution not normalized: {total} <= 0")
                            normalized = False

                    if not normalized:
                        raise ValueError("Distribution is not normalized.")

                    distrib_long = distrib_long.flatten()
                    distrib_short = distrib_short.flatten()

                    # Instead of checking for duplicates, generate all symmetries and add them
                    # We generate symmetries with both short and long paths, as the long path
                    # need not account for Bob input symmetries (accounted for by deterministic
                    # transformations), but is impacted by Alice's input symmetries.
                    computed_distributions.extend(
                        generate_distribution_symmetries_and_concatenate(
                            distrib_short=distrib_short,
                            distrib_long=distrib_long,
                        )
                    )

    return computed_distributions


# ################


if __name__ == "__main__":
    exp_name = str(input("Enter experiment name: ")).strip()
    exp_file = f"boxworld_extremals_{exp_name}.txt"

    for i in range(24):
        # Generate all extremal points of the routed Bell experiment boxworld strategies
        extremal_points = generate_all_extremal_points(
            delta=2, m_alice=2, m_bob_short=2, m_bob_long=2, idx_range=(i, i)
        )

        # Write the extremal points to a file
        with open(exp_file, "a") as f:
            for point in extremal_points:
                f.write(f"{point.tolist()}\n".replace("[", "").replace("]", "").replace(" ", ""))
                # IMPORTANT : it is okay to write duplicate points. We deduplicate before running the CDD solver.
            logger.success(f"State {i} extremal points written to {exp_file} successfully.")

    # Solve the boxworld polytope using the CDD solver
    from solve_full_polytope import PolytopeTypes, PolytopeWrapper, RepTypes

    # Load the vertices from the file
    seen = set()  # To deduplicate points
    vertices = []  # To store the vertices
    with open(exp_file, "r") as f:
        # Load the vertices from the file
        logger.info(f"Loading vertices from {exp_file}...")
        while True:
            line = f.readline()
            if not line:
                break
            # Convert the line to a numpy array and deduplicate
            point = np.array([float(x) for x in line.strip().split(",")])
            if point.tobytes() not in seen:
                seen.add(point.tobytes())
                vertices.append(point)
    # Convert the list of vertices to a numpy array
    vertices = np.array(vertices)
    # vertices = np.loadtxt(exp_file, delimiter=",")
    logger.info(f"Loaded {len(vertices)} unique vertices from boxworld_extremals.txt")

    v_representation = np.hstack(
        [
            np.ones(
                (len(vertices), 1)
            ),  # Add a column of ones to indicate these are proper vertices
            vertices,
        ]
    )

    # # Deduplicate vertices
    # v_representation_unique = np.unique(v_representation, axis=0)
    # print(f"Vertices shape (deduped): {v_representation_unique.shape}")
    # print(v_representation_unique[:, 0])  # Print first 5 vertices for debugging

    logger.info("Starting to solve the boxworld polytope...")
    solver = PolytopeWrapper(
        source_array=v_representation,
        rep_type=RepTypes.GENERATOR,
        lin_set=None,
    )
    logger.info("Boxworld polytope loaded.")

    solver.write_inequalities(
        polytope_type=PolytopeTypes.MEASURED,
        file_id=f"boxworld_polytope_{exp_name}",
    )

    bounded = solver.is_bounded()
    if bounded:
        logger.info("The boxworld polytope is bounded.")
    else:
        logger.warning("The boxworld polytope is unbounded?")
