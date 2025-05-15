from abc import ABC, abstractmethod

import numpy as np
from behaviors import Behavior
from loguru import logger


class Sampler(ABC):
    """
    Abstract base class for samplers.
    """

    @abstractmethod
    def sample(self) -> Behavior:
        pass


class UniformNormalizedSampler:
    """
    A sampler that generates uniformly distributed samples in a normalized space.
    """

    def __init__(self, delta: int, m: int, z: bool):
        """
        Initialize the sampler.
        """
        if delta != 2 or m != 2 or not z:
            logger.warning("Expect unexpected behaviors for delta =! 2 or m =! 2 or z = False")
        self.delta = delta
        self.m = m
        self.z = z
        self.np_sampling_shape = (self.z + 1, self.delta**2, self.m**2)

    def sample(self) -> Behavior:
        sampled = np.random.dirichlet(np.ones(self.delta**2), (self.z + 1) * self.m**2)

        if self.z:
            sample_s = sampled[: self.m**2, :].T
            sample_l = sampled[self.m**2 :, :].T
            sample = np.array([sample_s, sample_l])
            res = Behavior(sample)
        else:
            raise NotImplementedError("Sampling for z = False is not implemented.")

        return res


class NoSignalingSampler:
    pass
