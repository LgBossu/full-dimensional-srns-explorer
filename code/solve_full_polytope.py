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

    print(h_latent_matrix)
