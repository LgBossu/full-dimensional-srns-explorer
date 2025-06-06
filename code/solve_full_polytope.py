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

if __name__ == "__main__":
    delta = 2
    m = 2

    latent_set = no_signaling_sets.LatentSRNSSet(delta, m)

    # Get the H-representation of the latent set
    latent_matrix_form = latent_set.get_cdd_matrix()

    h_latent_matrix = cdd.Matrix(latent_matrix_form, number_type="float")

    latent_polytope = cdd.Polyhedron(h_latent_matrix)
    v_matrix = latent_polytope.get_generators()

    vertices = [list(v) for v in v_matrix]

    print(f"Number of vertices in the latent set: {len(vertices)}")
    print(f"vertex dimension: {len(vertices[0]) if vertices else 0}")

    actual_vertices = np.array(vertices)
    actual_vertices = actual_vertices[:, 1:]  # Remove the first column (the constant term)
    # Remove duplicate rows from the array
    actual_vertices = np.unique(actual_vertices, axis=0)
    print(f"Number of unique vertices: {len(actual_vertices)}")

    for v in actual_vertices[:10]:
        print(v)
