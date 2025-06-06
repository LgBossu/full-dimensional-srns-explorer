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
    A, b = latent_set.get_equations()

    max_mix_vector = behaviors.completely_mixed_behavior.get_vector()
    with np.printoptions(threshold=np.inf, linewidth=np.inf):  # type: ignore
        print("A:")
        print(A)
        print("b:")
        print(b)
        print("Maximally mixed vector:")
        print(max_mix_vector)
        print("A @ max_mix_vector:")
        print(A @ max_mix_vector)
        print("Equality check:")
        print(A @ max_mix_vector == b)
