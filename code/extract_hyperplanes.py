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
        samples: list[np.ndarray],
        belonging: list[bool] | bool = False,
    ) -> None:
        """
        Initializes the NonSRNSExtractor with the given parameters.

        :param delta: The delta value for the no-signaling set.
        :param m: The m value for the no-signaling set.
        :param samples: A list of samples (numpy arrays) to check against the no-signaling set.
        :param belonging: - A list of booleans indicating
        whether each sample belongs to the no-signaling set.
                          - If True, all samples are assumed to belong to the set.
                          - If False, belonging will be determined later.
        """
        self.delta = delta
        self.m = m
        self.samples = samples
        self.belonging = belonging

        if isinstance(belonging, list):
            if len(belonging) != len(samples):
                raise ValueError("The length of belonging must match the number of samples.")
            self.known_belonging = True
        elif isinstance(belonging, bool) and belonging:
            self.belonging = [True] * len(samples)
            self.known_belonging = True
        else:
            self.known_belonging = False

    def belongs(
        self,
        sample: np.ndarray | int,
        set: no_signaling_sets.ShortRangeNoSignalingSet,
    ) -> bool:
        if isinstance(sample, int):
            sample = self.samples[sample]
        if not isinstance(sample, np.ndarray):
            raise TypeError("Sample must be a numpy array or an integer index.")
        return set.is_in_set(sample)

    def determine_belonging(self):
        belonging_list = []
        srns_set = no_signaling_sets.ShortRangeNoSignalingSet(
            delta=self.delta,
            m=self.m,
        )

        logger.info("Determining belonging of samples to the SRNS set...")
        for sample in tqdm(self.samples):
            belonging_list.append(self.belongs(sample, srns_set))

        logger.info(f"Belonging determined for {len(belonging_list)} samples.")
        logger.warning("Overwriting previously set belonging values with computed values.")

        self.belonging = belonging_list
        self.known_belonging = True

    def extract(self) -> list[np.ndarray]:
        """From the provided samples and information, extracts
        only vectors that do NOT belong to SRNS."""

        if not self.known_belonging:
            logger.warning("Belonging is not known, determining belonging now.")
            self.determine_belonging()

        arr_samples = np.array(self.samples)
        arr_belonging = np.array(self.belonging, dtype=bool)

        return list(arr_samples[~arr_belonging])


class HyperplanesExtractor:
    def __init__(
        self,
        delta: int,
        m: int,
        samples: list[np.ndarray],
    ) -> None:
        """
        Initializes the HyperplanesExtractor with the given parameters.
        :param delta: The delta value for the no-signaling set.
        :param m: The m value for the no-signaling set.
        :param samples: A list of samples (numpy arrays) **assumed to NOT belong to the SRNS set**.
        """
        self.delta = delta
        self.m = m
        self.samples = samples
