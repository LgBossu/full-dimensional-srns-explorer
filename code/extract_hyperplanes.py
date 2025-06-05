from itertools import permutations

import behaviors
import no_signaling_sets
import numpy as np
from loguru import logger
from tqdm import tqdm


class NonSRNSExtractor:
    """A class to extract non-short-range no-signaling sets from
    arbitrary samples in NS."""

    def __init__(
        self,
        delta: int,
        m: int,
        arbitrary_samples: list[np.ndarray],
        belonging: list[bool] | bool = False,
    ) -> None:
        """
        Initializes the NonSRNSExtractor with the given parameters.

        :param delta: The delta value for the no-signaling set.
        :param m: The m value for the no-signaling set.
        :param samples: A list of samples (numpy arrays) to check against the no-signaling set.
        :param belonging: - A list of booleans indicating
        whether each sample belongs to the no-signaling set.
                          - **If True, all samples are assumed to already lie
                          OUTSIDE the set** (same as a list with only **FALSE**
                          values).
                          - If False, belonging will be determined later.
        """
        self.delta = delta
        self.m = m
        self.arbitrary_samples = arbitrary_samples
        self.belonging = belonging

        if isinstance(belonging, list):
            if len(belonging) != len(arbitrary_samples):
                raise ValueError("The length of belonging must match the number of samples.")
            self.known_belonging = True
        elif isinstance(belonging, bool) and belonging:
            self.belonging = [False] * len(arbitrary_samples)
            self.known_belonging = True
        else:
            self.known_belonging = False

    def belongs(
        self,
        sample: np.ndarray | int,
        srns_set: no_signaling_sets.ShortRangeNoSignalingSet,
    ) -> bool:
        if isinstance(sample, int):
            sample = self.arbitrary_samples[sample]
        if not isinstance(sample, np.ndarray):
            raise TypeError("Sample must be a numpy array or an integer index.")
        return srns_set.is_in_set(sample)

    def determine_belonging(
        self,
        srns_set: no_signaling_sets.ShortRangeNoSignalingSet | None = None,
    ) -> None:
        belonging_list = []
        if srns_set is None:
            srns_set = no_signaling_sets.ShortRangeNoSignalingSet(
                delta=self.delta,
                m=self.m,
            )

        logger.info("Determining belonging of samples to the SRNS set...")
        for sample in tqdm(self.arbitrary_samples):
            belonging_list.append(self.belongs(sample, srns_set))

        logger.info(f"Belonging determined for {len(belonging_list)} samples.")
        logger.warning("Overwriting previously set belonging values with computed values.")

        self.belonging = belonging_list
        self.known_belonging = True

    def extract_non_srns(self) -> list[np.ndarray]:
        """From the provided samples and information, extracts
        only vectors that do NOT belong to SRNS."""

        if not self.known_belonging:
            logger.warning("Belonging is not known, determining belonging now.")
            self.determine_belonging()

        arr_samples = np.array(self.arbitrary_samples)
        arr_belonging = np.array(self.belonging, dtype=bool)

        return list(arr_samples[~arr_belonging])


class HyperplanesExtractor:
    """
    A class to extract hyperplanes from samples that do not belong to the
    Short-Range No-Signaling Set.
    """

    def __init__(
        self,
        delta: int,
        m: int,
        non_srns_samples: list[np.ndarray],
    ) -> None:
        """
        Initializes the HyperplanesExtractor with the given parameters.

        :param delta: The delta value for the no-signaling set.
        :param m: The m value for the no-signaling set.
        :param samples: A list of samples (numpy arrays) **assumed to NOT belong to the SRNS set**.
        """
        self.delta = delta
        self.m = m
        self.non_srns_samples = non_srns_samples

    # UTILS
    def scale_down_vector(
        self,
        vector: np.ndarray,
        atol: float = 1e-10,
    ) -> np.ndarray:
        """
        If all elements are equal to a common factor up to sign
        changes, we rescale the vector by that factor to make
        it a vector of integers {-1, 0, +1}.

        - Raises a **ValueError** if the vector cannot be scaled down uniformly.
        - Raises a **ZeroDivisionError** on the zero vector (or vectors close to it).

        - **Else, returns the rescaled vector as an integer array.**

        *(It just so happened experimentally that in low
        dimensions, this rescaling can be done for all
        hyperplanes somehow.)*
        """
        mask = np.abs(vector) > atol
        # Mask to work on the non-zero elements

        if not np.any(mask):
            raise ZeroDivisionError(
                "Vector cannot be scaled down uniformly: all elements are zero or close to."
            )

        factor = np.abs(vector[mask])[0]

        abs_is_cst = np.allclose(
            np.abs(vector[mask]),
            factor,
            atol=atol,
        )  # Check that all non-zero elements are equal to the factor

        if abs_is_cst:
            # Divide by the factor and round to nearest integer to avoid floating point issues
            rescaled = np.zeros_like(vector)
            rescaled[mask] = np.round(vector[mask] / factor).astype(int)
            return rescaled.astype(int)
        else:
            raise ValueError(f"Vector cannot be scaled down uniformly : {vector}")

    def normalize_vector(
        self,
        vector: np.ndarray,
    ) -> np.ndarray:
        """
        Normalizes the vector to have unit length.
        If the vector is close to zero, it returns the zero vector.
        """
        if np.allclose(vector, 0, atol=1e-10):
            return vector
        else:
            return vector / np.linalg.norm(vector)

    # HYPERPLANE EXTRACTION
    def extract_from_vector(
        self,
        vector: np.ndarray,
        srns_set: no_signaling_sets.ShortRangeNoSignalingSet,
    ) -> np.ndarray:
        # Get the hyperplane equation from the vector
        hyperplane_eq = srns_set.get_facet_hyperplane(
            behaviors.RoutedBehavior(self.delta, self.m, vector)
        )

        # Scale down the hyperplane equation to have integer coefficients
        try:
            hyperplane_eq = self.scale_down_vector(hyperplane_eq)
        except ValueError:
            logger.warning(f"Vector {vector} cannot be scaled down uniformly, normalizing instead.")
            hyperplane_eq = self.normalize_vector(hyperplane_eq)
        except ZeroDivisionError as e:
            logger.error("Vector is zero or close to zero, should not be a hyperplane.")
            raise e

        return hyperplane_eq

    def extract_hyperplanes(
        self,
        srns_set: no_signaling_sets.ShortRangeNoSignalingSet | None = None,
    ) -> list[np.ndarray]:
        """
        Extracts hyperplanes from the provided samples.

        :param srns_set: The ShortRangeNoSignalingSet to use for extracting
        hyperplanes. Automatically created if None.
        :return: A list of hyperplane equations (numpy arrays).
        """
        if srns_set is None:
            srns_set = no_signaling_sets.ShortRangeNoSignalingSet(
                delta=self.delta,
                m=self.m,
            )
        hyperplanes = []
        logger.info("Extracting hyperplanes from samples...")
        for sample in tqdm(self.non_srns_samples):
            try:
                hyperplane_eq = self.extract_from_vector(sample, srns_set)
                hyperplanes.append(hyperplane_eq)
            except ZeroDivisionError as e:
                logger.error(f"Skipping zero vector: {sample}. Error: {e}")
                hyperplanes.append(None)  # Append None for index consistency

        return hyperplanes


class QuotientHyperplanes:
    def __init__(self, delta: int, m: int, hyperplanes: list[np.ndarray]) -> None:
        """
        Initializes the QuotientHyperplanes with the given parameters.

        :param delta: The delta value for the no-signaling set.
        :param m: The m value for the no-signaling set.
        :param hyperplanes: A list of hyperplane equations (numpy arrays).
        """
        self.delta = delta
        self.m = m
        self.hyperplanes = hyperplanes

    # SIGN CONSISTENCY
    def canonicalize(eq: np.ndarray) -> np.ndarray:
        """
        Canonicalizes a hypeperplane equation by flattening its array if needed,
        and setting the first nonzero element to be positive.

        This enables consistent representation of hyperplanes, for later
        comparison in sets.

        Note that :
        - This may switch the sign of the hyperplane equation (ie, the normal vector's direction).
        - This does not guarantee that the hyperplane is normalized (ie, unit length).
        - This returns a 1D numpy array.
        """
        flat = eq.flatten()
        idx = np.flatnonzero(flat)
        if idx.size and flat[idx[0]] < 0:
            flat = -flat
        return flat

    # FORMATTING FOR PERMUTATIONS
    def format_hyperplane(
        self,
        equation: np.ndarray,
    ) -> tuple[np.ndarray]:
        # TODO : ensure the function is applied to canonicalized hyperplanes.

        el_S, el_L, el_NS = (
            equation[0 : self.delta**2 * self.m**2],
            equation[self.delta**2 * self.m**2 : 2 * self.delta**2 * self.m**2],
            equation[2 * self.delta**2 * self.m**2 :],
        )
        el_S = el_S.reshape((self.delta, self.delta, self.m, self.m))
        el_L = el_L.reshape((self.delta, self.delta, self.m, self.m))
        el_NS = el_NS.reshape((self.delta,) * self.m)
        return (el_S, el_L, el_NS)

    def format_list_of_hyperplanes(
        self,
        equations: list[list[np.ndarray]],
    ) -> list[tuple[np.ndarray]]:
        formatted: list[tuple[np.ndarray]] = []
        for el in equations:
            formatted.append(self.format_hyperplane(el))

        return formatted

    def flatten_hyperplane(
        self,
        hyperplane: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        el_S, el_L, el_NS = hyperplane

        equation = np.concatenate((el_S.flatten(), el_L.flatten(), el_NS.flatten()))

        return equation

    def flatten_list_of_hyperplanes(
        self,
        hyperplanes: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        flattened = []
        for hyperplane in hyperplanes:
            flattened.append(self.flatten_hyperplane(hyperplane))
        return np.array(flattened, dtype=int)

    # PERMUTATIONS
    def permute_a(
        self,
        formatted_hyperplane: tuple[np.ndarray],
    ) -> list[tuple[np.ndarray]]:
        """Apply to a properly formatted hyperplane the permutation of coordinates
        corresponding to relabelings of a values."""
        permuted_hyperplanes = []
        for perm in permutations(range(self.delta)):
            permuted_S = formatted_hyperplane[0][list(perm), :, :, :]
            permuted_L = formatted_hyperplane[1][list(perm), :, :, :]
            permuted_NS = formatted_hyperplane[2]

            permuted_hyperplanes.append((permuted_S, permuted_L, permuted_NS))
        return permuted_hyperplanes

    def permute_b(
        self,
        formatted_hyperplane: tuple[np.ndarray],
    ) -> list[tuple[np.ndarray]]:
        """Apply to a properly formatted hyperplane the permutation of coordinates
        corresponding to relabelings of b values."""
        permuted_hyperplanes = []
        for perm in permutations(range(self.delta)):
            permuted_S = formatted_hyperplane[0][:, list(perm), :, :]
            permuted_L = formatted_hyperplane[1][:, list(perm), :, :]
            permuted_NS = formatted_hyperplane[2][np.ix_(*([list(perm)] * self.m))]

            permuted_hyperplanes.append((permuted_S, permuted_L, permuted_NS))
        return permuted_hyperplanes

    def permute_x(
        self,
        formatted_hyperplane: tuple[np.ndarray],
    ) -> list[tuple[np.ndarray]]:
        """Apply to a properly formatted hyperplane the permutation of coordinates
        corresponding to relabelings of x values."""
        permuted_hyperplanes = []
        for perm in permutations(range(self.m)):
            permuted_S = formatted_hyperplane[0][:, :, list(perm), :]
            permuted_L = formatted_hyperplane[1][:, :, list(perm), :]
            permuted_NS = formatted_hyperplane[2]

            permuted_hyperplanes.append((permuted_S, permuted_L, permuted_NS))
        return permuted_hyperplanes

    def permute_y(
        self,
        formatted_hyperplane: tuple[np.ndarray],
    ) -> list[tuple[np.ndarray]]:
        """Apply to a properly formatted hyperplane the permutation of coordinates
        corresponding to relabelings of y values."""
        permuted_hyperplanes = []
        for perm in permutations(range(self.m)):
            permuted_S = formatted_hyperplane[0][:, :, :, list(perm)]
            permuted_L = formatted_hyperplane[1][:, :, :, list(perm)]
            permuted_NS = formatted_hyperplane[2]

            permuted_hyperplanes.append((permuted_S, permuted_L, permuted_NS))
        return permuted_hyperplanes

    list_of_permutations = [permute_a, permute_b, permute_x, permute_y]

    def permute(
        self,
        axis: int | str,
        hyperplane: np.ndarray,
    ) -> list[np.ndarray]:
        """Permute a hyperplane along the specified axis."""
        formatted_hyperplane = self.format_hyperplane(hyperplane)
        # TODO : where do i canonicalize the hyperplane?

        if isinstance(axis, str):
            axis = axis.lower()
            if axis == "a":
                unformatted_res = self.permute_a(formatted_hyperplane)
            elif axis == "b":
                unformatted_res = self.permute_b(formatted_hyperplane)
            elif axis == "x":
                unformatted_res = self.permute_x(formatted_hyperplane)
            elif axis == "y":
                unformatted_res = self.permute_y(formatted_hyperplane)
            else:
                raise ValueError(f"Unknown axis: {axis}")
        elif isinstance(axis, int):
            if 0 <= axis < 4:
                unformatted_res = self.list_of_permutations[axis](formatted_hyperplane)
            else:
                raise ValueError(f"Axis index out of range: {axis}")
        else:
            raise TypeError(f"Axis must be an int or a str, got {type(axis)}")

        return self.flatten_list_of_hyperplanes(unformatted_res)

    # TODO : continue class programming if relevant
    # Refer to locally stored (or in previous versions)
    # jupyter notebook regarding implementation of hyperplanes
    # equivalence under permutations and quotienting.
