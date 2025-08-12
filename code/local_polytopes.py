"""
To give ourselves a proper comparison in higher-dimensional studies i.e. of boxworld,
we want to solve the local polytope for varying input/output numbers.
"""

import itertools

import numpy as np
from loguru import logger

n_inputs_alice = 2
n_inputs_bob = 3
n_outputs = 2


if __name__ == "__main__":
    # Generate all local deterministic vertices as dense arrays of shape
    # (nO, nO, nA, nB), where P[a,b,x,y] is 1.0 at a=(A_map[x]), b=(B_map[y]).
    A_strats = list(itertools.product(range(n_outputs), repeat=n_inputs_alice))
    B_strats = list(itertools.product(range(n_outputs), repeat=n_inputs_bob))

    vertices: list[np.ndarray] = []

    for A_map in A_strats:
        for B_map in B_strats:
            # print(f"A_map: {A_map}, B_map: {B_map}")
            P = np.zeros((n_outputs, n_outputs, n_inputs_alice, n_inputs_bob), dtype=float)
            for x in range(n_inputs_alice):
                a = A_map[x]
                for y in range(n_inputs_bob):
                    b = B_map[y]
                    P[a, b, x, y] = 1.0
            vertices.append(P.flatten())

    # Save the vertices to a file
    with open("local_polytope_vertices.npy", "w") as f:
        for vertex in vertices:
            f.write(f"{vertex.tolist()}\n".replace("[", "").replace("]", "").replace(" ", ""))

    # Solve the local polytope
    from solve_full_polytope import PolytopeTypes, PolytopeWrapper, RepTypes

    vertices_as_array = np.array(vertices, dtype=float)
    v_representation = np.hstack(
        [
            np.ones(
                (len(vertices), 1)
            ),  # Add a column of ones to indicate these are proper vertices
            vertices,
        ]
    )

    logger.info("Starting to solve the local polytope...")
    solver = PolytopeWrapper(
        source_array=v_representation,
        rep_type=RepTypes.GENERATOR,
        lin_set=None,
    )
    logger.info("Local polytope loaded.")

    solver.write_inequalities(
        polytope_type=PolytopeTypes.MEASURED,
        file_id=f"local_polytope_{(n_inputs_alice, n_inputs_bob, n_outputs)}",
    )

    bounded = solver.is_bounded()
    if bounded:
        logger.info("The local polytope is bounded.")
    else:
        logger.warning("The local polytope is unbounded?")
