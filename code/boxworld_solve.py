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

from enum import Enum
from time import time
from typing import Union

import numpy as np
from behaviors import routed_indices_to_index, short_range_indices_to_index
from loguru import logger


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


complements = {
    BoxworldEffect.e1: BoxworldEffect.e3,
    BoxworldEffect.e2: BoxworldEffect.e4,
    BoxworldEffect.e3: BoxworldEffect.e1,
    BoxworldEffect.e4: BoxworldEffect.e2,
    BoxworldEffect.u: BoxworldEffect.zero,
    BoxworldEffect.zero: BoxworldEffect.u,
}


def get_complement_effect(effect: BoxworldEffect) -> BoxworldEffect:
    """
    Get the complement effect of a given boxworld effect, to properly define binary measurements.

    We define the complement effect such (1/2)*(e + e') = u, where u is the unit effect.

    We actually just have u~u, e1~e3, e2~e4.
    """
    return complements[effect]


def generate_all_extremal_points() -> list[np.ndarray]:
    """
    Generate all extremal points of the routed Bell experiment boxworld strategies.
    """

    computed_distributions = []  # This will hold all computed effects
    computed_total = 0
    start_time = time()

    for state in BoxworldBipartiteVertex:
        logger.info(f"Processing state: {state.value.arr}")
        for (
            alice_measurement0,
            alice_measurement1,
            bob_short_measurement0,
            bob_short_measurement1,
        ) in [
            (i, j, k, l)
            for i in BoxworldEffect
            for j in BoxworldEffect
            for k in BoxworldEffect
            for l in BoxworldEffect
        ]:
            if alice_measurement0 == alice_measurement1:
                continue  # Skip if Alice's measurements are the same, this is a trivial strategy
            alice_complement0 = get_complement_effect(alice_measurement0)
            alice_complement1 = get_complement_effect(alice_measurement1)
            if bob_short_measurement0 == bob_short_measurement1:
                continue  # Skip if Bob's short channel measurements are the same, this is a trivial strategy
            bob_short_complement0 = get_complement_effect(bob_short_measurement0)
            bob_short_complement1 = get_complement_effect(bob_short_measurement1)

            alice = [
                [alice_measurement0.value.arr, alice_complement0.value.arr],
                [alice_measurement1.value.arr, alice_complement1.value.arr],
            ]
            bob_short = [
                [bob_short_measurement0.value.arr, bob_short_complement0.value.arr],
                [bob_short_measurement1.value.arr, bob_short_complement1.value.arr],
            ]

            for bob_long_measurement in BoxworldEffect:
                bob_long_complement = get_complement_effect(bob_long_measurement)
                bob_long = [bob_long_measurement.value.arr, bob_long_complement.value.arr]

                for transform in TransformOutput:  # We pick bob's deterministic transformation from the degraded state to the final output
                    # Compute the output probabilities for the current configuration
                    distribution = np.zeros(32, dtype=float)

                    # Compute the short-path distribution
                    # logger.debug("Computing short-path distribution")
                    for a, b, x, y in [
                        (i, j, k, l) for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]
                    ]:
                        distribution[routed_indices_to_index(a, b, x, y, 0)] = np.dot(
                            np.kron(alice[x][a], bob_short[y][b]), state.value.arr
                        )

                    # Compute the long-path distribution
                    # logger.debug("Computing long-path distribution")
                    for a, b, x, y in [
                        (i, j, k, l) for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]
                    ]:
                        for b_prime in [0, 1]:
                            distribution[
                                routed_indices_to_index(a, transform.value.arr[b_prime, y], x, y, 1)
                            ] += np.dot(np.kron(alice[x][a], bob_long[b_prime]), state.value.arr)

                    # Check normalization on the short-path
                    # it should already be normalized
                    normalized = True
                    for x, y in [(i, j) for i in [0, 1] for j in [0, 1]]:
                        total = np.sum(
                            [
                                distribution[i]
                                for i in [
                                    routed_indices_to_index(_a, _b, x, y, 0)
                                    for _a in [0, 1]
                                    for _b in [0, 1]
                                ]
                            ]
                        )
                        if total != 1:
                            logger.error(f"Distribution not normalized: {total} != 1")
                            normalized = False

                    # Normalize the long-path distribution
                    for x in [0, 1]:
                        total = np.sum(
                            [
                                distribution[i]
                                for i in [
                                    short_range_indices_to_index((_a, _beta, x, 1))
                                    for _a in [0, 1]
                                    for _beta in [
                                        (beta0, beta1) for beta0 in [0, 1] for beta1 in [0, 1]
                                    ]
                                ]
                            ]
                        )
                        if total > 0:
                            for i in [
                                short_range_indices_to_index((_a, _beta, x, 1))
                                for _a in [0, 1]
                                for _beta in [
                                    (beta0, beta1) for beta0 in [0, 1] for beta1 in [0, 1]
                                ]
                            ]:
                                distribution[i] /= total
                        else:
                            raise ValueError(f"Zero probability along path (x={x})")

                    # Log info periodically
                    computed_total += 1
                    if computed_total % 10000 == 0:
                        logger.info(f"Computed {computed_total}/1.200.000 distributions so far.")
                        elapsed_time = time() - start_time
                        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
                        eta = (time() - start_time) * (1_200_000 - computed_total) / computed_total
                        logger.info(f"Estimated time remaining: {eta//60} minutes")

                    # Flatten the distribution to a 1D array
                    distribution = distribution.flatten()

                    # Check if this distribution has already been computed
                    if np.any([np.array_equal(distribution, d) for d in computed_distributions]):
                        pass
                    else:
                        computed_distributions.append(distribution)
                        logger.success(f"Added distribution:\n{distribution.reshape((2,4,4))}")
                        if not normalized:
                            logger.error("Distribution is not normalized.")
                            logger.info(
                                f"\nstate:   {state.value.arr}, \n"
                                f"alice:     {alice}, \n"
                                f"bob_short: {bob_short}, \n"
                                f"bob_long:  {bob_long}, \n"
                                f"transform: {transform.value.arr}\n"
                            )
                            raise ValueError("Distribution is not normalized.")

    return computed_distributions


if __name__ == "__main__":
    # Generate all extremal points of the routed Bell experiment boxworld strategies
    extremal_points = generate_all_extremal_points()

    # Print the number of unique extremal points found
    print(f"Number of unique extremal points: {len(extremal_points)}")

    # Write the extremal points to a file
    with open("boxworld_extremals.txt", "w") as f:
        for point in extremal_points:
            f.write(f"{point.tolist()}\n".replace("[", "").replace("]", ""))
