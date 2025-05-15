"""This module aims to provide in matrix form the equations defining the no signaling set."""

import behaviors
import numpy as np
from loguru import logger


def routed_no_signaling_equations(delta: int, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the no-signaling equations for the routed case.
    """
    # Polytope dimension
    dim = 2 * (delta - 1) * m + (delta - 1) ** 2 * m**2
    logger.debug(f"Polytope dimension: {dim}")

    # Initialize the equations
    equations = []
    right_side = []

    # Normalization equations
    for x, y, z in [(i, j, k) for i in range(m) for j in range(m) for k in [0, 1]]:
        eq = np.zeros(2 * delta**2 * m**2)
        for a, b in [(i, j) for i in range(delta) for j in range(delta)]:
            eq[behaviors.routed_indices_to_index(a, b, x, y, z, delta, m)] = 1
        equations.append(eq)
        right_side.append(1)
        logger.debug(f"Normalization equation: {eq} = 1, xyz = {x, y, z}")

    # No-signaling equations
    # From Alice to Bob
    for b, y, z in [(i, j, k) for i in range(delta) for j in range(m) for k in [0, 1]]:
        for x in range(m):
            eq = np.zeros(2 * delta**2 * m**2)
            if x == 0:
                continue
            else:
                for a in range(delta):
                    eq[behaviors.routed_indices_to_index(a, b, x, y, z, delta, m)] = -1
                    eq[behaviors.routed_indices_to_index(a, b, 0, y, z, delta, m)] = 1
                equations.append(eq)
                right_side.append(0)
                logger.debug(f"No-signaling equation: {eq} = 0, bxyz = {b, x, y, z}")
    # From Bob to Alice
    for a, x in [(i, j) for i in range(delta) for j in range(m)]:
        for y, z in [(j, k) for j in range(m) for k in [0, 1]]:
            eq = np.zeros(2 * delta**2 * m**2)
            if (y, z) == (0, 0):
                continue
            else:
                for b in range(delta):
                    eq[behaviors.routed_indices_to_index(a, b, x, y, z, delta, m)] = -1
                    eq[behaviors.routed_indices_to_index(a, b, x, 0, 0, delta, m)] = 1
                equations.append(eq)
                right_side.append(0)
                logger.debug(f"No-signaling equation: {eq} = 0, axyz = {a, x, y, z}")

    # Convert to numpy arrays
    equations = np.array(equations)
    right_side = np.array(right_side)

    # TODO : reduce the rank of the equations matrix to equal dim(B) - dim(NS) or smth like that

    # Log the shapes and values of the equations and right side
    logger.debug(f"Delta: {delta}, m: {m}")
    logger.debug(f"Equations shape: {equations.shape}")
    logger.debug(f"Equations: {equations}")
    logger.debug(f"Right side shape: {right_side.shape}")
    logger.debug(f"Right side: {right_side}")

    return equations, right_side, dim
