"""
A script to get the complete polytope representation of the SRNS
set, ultimately in the experiment space.
Performs vertex enumeration on the latent SRNS set's space,
projects the vertices onto the experiment space, and then
computes the hyperplane representation of the resulting polytope.
"""

import behaviors
import cdd
import no_signaling_sets
import numpy as np
from loguru import logger

if __name__ == "__main__":
    delta = 2
    m = 2

    # Output results to a file
    latent_output_file = "output/latent_polytope_vertices.txt"
    measured_output_file = "output/measured_polytope_vertices.txt"
    measured_H_representation_file = "output/measured_polytope_H_representation.txt"

    logger.info(f"Computing the latent SRNS set for delta={delta} and m={m}...")
    latent_set = no_signaling_sets.LatentSRNSSet(delta, m)

    # Get the H-representation of the latent set
    logger.info("Computing the latent SRNS set's H-representation...")
    latent_matrix_form, lin_list = latent_set.get_cdd_matrix()

    logger.info(f"Setting the latent set's linear set to {len(lin_list)} linear constraints.")
    h_latent_matrix = cdd.Matrix(latent_matrix_form, number_type="float")
    lin_set = set(lin_list)
    h_latent_matrix.lin_set = lin_set

    logger.info(f"Computing the latent SRNS set's polytope with {len(lin_set)} linear constraints.")
    latent_polytope = cdd.Polyhedron(h_latent_matrix)
    v_matrix = latent_polytope.get_generators()

    vertices = [list(v) for v in v_matrix]

    print(f"Number of vertices in the latent set: {len(vertices)}")
    print(f"vertex dimension: {len(vertices[0]) if vertices else 0}")

    logger.info(f"Writing the latent SRNS set's vertices to {latent_output_file}...")
    with open(latent_output_file, "w") as f:
        f.write(
            f"Vertex dimension: {len(vertices[0]) if vertices else "error: no vertex"}\n"  # noqa: E501
        )
        f.write(f"{len(vertices)} vertices found\n\n")
        for v in vertices:
            f.write(" ".join(map(str, v)) + "\n")

    # Project the vertices onto the experiment space
    logger.info("Projecting the latent SRNS set's vertices onto the experiment space...")
    projection_matrix = latent_set.latent_to_measured()
    projection_matrix = np.hstack(
        (projection_matrix, np.zeros((projection_matrix.shape[0], 1)))
    )  # Add a column of zeros for the last dimension

    measured_vertices = np.dot(vertices, projection_matrix.T)

    # Set up the measured polytope
    logger.info("Setting up the measured polytope...")
    measured_matrix_form = cdd.Matrix(measured_vertices, number_type="float")
    measured_matrix_form.rep_type = cdd.RepType.GENERATOR
    measured_polytope = cdd.Polyhedron(measured_matrix_form)

    measured_v_matrix = measured_polytope.get_generators()

    measured_vertices = [list(v) for v in measured_v_matrix]
    measured_vertices = np.array(measured_vertices)
    measured_vertices = measured_vertices
    measured_vertices = np.unique(measured_vertices, axis=0)

    print(
        f"Dimension of the vertices in the measured set: {len(measured_vertices[0]) if measured_vertices.size else 'error: no vertex'}"  # noqa: E501
    )
    print(f"Number of vertices in the measured set: {len(measured_vertices)}")

    logger.info(f"Writing the measured polytope's vertices to {measured_output_file}...")
    with open(measured_output_file, "w") as f:
        f.write(
            f"Vertex dimension: {len(measured_vertices[0]) if measured_vertices.size else 'error: no vertex'}\n"  # noqa: E501
        )
        f.write(f"{len(measured_vertices)} vertices found\n\n")
        for v in measured_vertices:
            f.write(" ".join(map(str, v)) + "\n")
    # Compute the H-representation of the measured polytope
    logger.info("Computing the H-representation of the measured polytope...")
    measured_h_matrix = measured_polytope.get_inequalities()
    measured_lin_set = measured_h_matrix.lin_set

    logger.info(
        f"Writing the H-representation of the measured polytope to {measured_H_representation_file}..."  # noqa: E501
    )
    # Write the H-representation to a file
    with open(measured_H_representation_file, "w") as f:
        f.write(f"Linear set: {measured_lin_set}\n")
        f.write(f"{len(measured_h_matrix)} inequalities\n")
        for h in measured_h_matrix:
            f.write(" ".join(map(str, h)) + "\n")
    print(f"Number of inequalities in the measured set: {len(measured_h_matrix)}")

    # TODO : sanity check, do our SRNS measured behaviors match the representation?

    logger.info("Done computing the polytope representations.")
    logger.info("You can now use the output files for further analysis.")
