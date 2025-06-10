"""
A script to get the complete polytope representation of the SRNS
set, ultimately in the experiment space.
Performs vertex enumeration on the latent SRNS set's space,
projects the vertices onto the experiment space, and then
computes the hyperplane representation of the resulting polytope.
"""

if __name__ == "__main__":
    # Import necessary libraries
    import cdd
    import no_signaling_sets
    import numpy as np
    from loguru import logger

    # Dimensions and parameters
    delta = 2
    m = 2

    # Output results to a file
    latent_output_file = "output/latent_polytope_vertices.txt"
    measured_output_file = "output/measured_polytope_vertices.txt"
    measured_H_representation_file = "output/measured_polytope_H_representation.txt"

    # Set up the Set object to get the equations
    logger.info(f"Computing the latent SRNS set for delta={delta} and m={m}...")
    latent_set = no_signaling_sets.LatentSRNSSet(delta, m)

    # Get the H-representation of the latent set
    logger.info("Computing the latent SRNS set's H-representation...")
    latent_matrix_form, lin_list = latent_set.get_cdd_matrix()

    # Convert the latent set's H-representation to a cdd.Matrix
    logger.info(f"Setting the latent set's linear set to {len(lin_list)} linear constraints.")
    h_latent_matrix = cdd.Matrix(latent_matrix_form, number_type="float")
    lin_set = set(lin_list)
    h_latent_matrix.lin_set = lin_set  # Label the equations as such, using the linear set
    h_latent_matrix.rep_type = cdd.RepType.INEQUALITY  # Set the representation type to inequality

    # # Reduce dimension
    # logger.warning(h_latent_matrix.canonicalize())

    # Compute the double description of the latent set's polytope
    logger.info(f"Computing the latent SRNS set's polytope with {len(lin_set)} linear constraints.")
    latent_polytope = cdd.Polyhedron(h_latent_matrix)

    # Get the vertices of the latent polytope
    v_matrix = latent_polytope.get_generators()
    # logger.warning(v_matrix)
    vertices = [list(v) for v in v_matrix]

    # Log some information to the user
    logger.info(f"Number of vertices in the latent set: {len(vertices)}")
    logger.info(f"vertex dimension: {len(vertices[0]) if vertices else 0}")

    # Safe computed information to a file
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
    # Get the projection matrix, which maps vertices from the latent space to the measured space
    # The matrix does NOT account for Y coordinates, we must add a column of zeros
    projection_matrix = latent_set.latent_to_measured()
    projected_generators = []

    for row in v_matrix:  # v_matrix from get_generators()
        tag = row[0]  # 0 or 1
        vec = np.array(row[1:], dtype=float)  # The actual latent vector

        projected_vec = projection_matrix @ vec  # shape (d_measured,)
        new_row = [tag] + projected_vec.tolist()

        projected_generators.append(new_row)

    measured_vertices = np.array(projected_generators, dtype=float)

    # Set up the measured polytope with the projected generators
    logger.info("Setting up the measured polytope...")
    measured_matrix_form = cdd.Matrix(
        measured_vertices, number_type="float"
    )  # Do we need a linear set here?
    measured_matrix_form.rep_type = cdd.RepType.GENERATOR
    measured_polytope = cdd.Polyhedron(measured_matrix_form)

    # Get the vertices of the measured polytope
    measured_v_matrix = measured_polytope.get_generators()
    measured_vertices = [list(v) for v in measured_v_matrix]
    measured_vertices = np.array(measured_vertices, dtype=float)

    # Log some information about the measured polytope
    logger.info(
        f"Dimension of the vertices in the measured set: {len(measured_vertices[0]) if measured_vertices.size else 'error: no vertex'}"  # noqa: E501
    )
    logger.info(f"Number of vertices in the measured set: {len(measured_vertices)}")

    # Write the measured polytope's vertices to a file
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

    # Write the H-representation to a file
    logger.info(
        f"Writing the H-representation of the measured polytope to {measured_H_representation_file}..."  # noqa: E501
    )
    with open(measured_H_representation_file, "w") as f:
        f.write(f"Linear set: {measured_lin_set}\n")
        f.write(f"{len(measured_h_matrix)} inequalities\n")
        for h in measured_h_matrix:
            f.write(" ".join(map(str, h)) + "\n")
    logger.info(f"Number of inequalities in the measured set: {len(measured_h_matrix)}")

    # TODO : sanity check, do our SRNS measured behaviors match the representation?
    check_files_suffix = "delta_2_m_2_samples_2000000_runtime_1747736589.6014528"
    samples = np.load(
        f"data/view_srns/sampled_behaviors_{check_files_suffix}.npy"
    )  # This loads measured behaviors
    belongings = np.load(f"data/view_srns/belonging_list_{check_files_suffix}.npy")
    # A boolean list indicating if the sampled behaviors belong to the measured polytope
    logger.info("Checking if the sampled behaviors belong to the measured polytope...")

    # HINT : our inferred belongings boolean list should match the belonging list

    logger.info("Done computing the polytope representations.")
    logger.info("You can now use the output files for further analysis.")
