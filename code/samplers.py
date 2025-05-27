"""
This module defines some sampling tools, that wrap around
methods to generate points (behavior instances) to study.

Notably, the `NoSignalingSampler` class can be instanciated on
any values of `delta` and `m` (proper to the experimental
setting) and, using the polytopewalk library [see references and
licence], generates uniformly distributed samples in the
no-signaling set of behaviors.

These samples can be used for computational tasks, such as
volume estimation of SRNS within NS, or hyperplanes search.

The SamplesAnalyzer class aims to provide a few tools to
analyze a sampled distribution, notably by plotting
projections or computing statistics on the distribution.
"""

import sys
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import polytopewalk as pw
import scipy as sp
from behaviors import RoutedBehavior
from loguru import logger
from no_signaling_sets import NoSignalingSet
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Sampler(ABC):
    """
    Abstract base class for samplers.
    """

    @abstractmethod
    def sample(self) -> RoutedBehavior:
        pass


class NoSignalingSampler(Sampler):
    def __init__(self, delta: int, m: int, z: bool = True):
        """
        Initialize the sampler.
        """
        if not z:
            raise NotImplementedError("Sampling for z = False is not implemented.")
        self.delta = delta
        self.m = m
        self.z = z

    def sample_multiple(
        self,
        number_of_samples: int = int(1e5),
        number_to_burn: int = int(1e3),
    ) -> np.ndarray:
        # Get the equations of the no-signaling set
        logger.trace("Getting the equations of the no-signaling set")
        ns_set = NoSignalingSet(delta=self.delta, m=self.m)
        A, b = ns_set.get_equations()

        logger.trace(f"Equations shape: {A.shape}")

        # Rank reduce the matrix
        logger.trace("Rank reducing the matrix")
        U, s, _ = sp.linalg.svd(A)
        rank = np.sum(s > 1e-10)
        A_reduced = U[:, :rank].T @ A
        b_reduced = U[:, :rank].T @ b

        # Change to sparse representation
        logger.trace("Changing to sparse representation")
        A_comp = sp.sparse.csc_matrix(A_reduced, dtype=np.float64)
        b_comp = b_reduced.astype(np.float64)

        # Get the polytopewalk objects
        logger.trace("Getting the polytopewalk objects")
        # Walk
        logger.trace("Getting the walk object")
        walk = pw.sparse.SparseHitAndRun()
        # Facial reduction
        logger.trace("Getting the facial reduction object")
        fr = pw.FacialReduction()
        fr_output = fr.reduce(A_comp, b_comp, k=A_comp.shape[1], sparse=True)
        # Center
        logger.trace("Getting the center object")
        sc = pw.sparse.SparseCenter()
        logger.trace(
            f"Test center point: {sc.getInitialPoint(fr_output.sparse_A, fr_output.sparse_b, A_comp.shape[1])}"  # noqa: E501
        )

        # # Run the MCMC
        logger.trace("Running the MCMC")
        samples_comp = pw.sparseFullWalkRun(
            A=fr_output.sparse_A,
            b=fr_output.sparse_b,
            k=A_comp.shape[1],
            num_sim=number_of_samples,
            walk=walk,
            fr=fr,
            sc=sc,
            burn=number_to_burn,
        )

        # Map back to the original space
        logger.trace("Mapping back to the original space")
        if fr_output.Q is not None and fr_output.Q.size > 0:
            logger.trace(f"Mapping back to the original space with Q: {fr_output.Q}")
            samples_og = (fr_output.Q @ samples_comp.T).T + fr_output.z1.T
        else:
            logger.trace("Already in the original space, no mapping needed")
            samples_og = samples_comp  # Already in original space
        logger.success(f"Samples shape: {samples_og.shape}")

        return samples_og

    def sample(self) -> RoutedBehavior:
        return RoutedBehavior(
            delta=self.delta,
            m=self.m,
            vector=self.sample_multiple(number_of_samples=1, number_to_burn=500)[0],
        )


class SamplesAnalyzer:
    """
    A class to compute statistics on samples distributions.
    """

    def __init__(self, delta: int, m: int, samples: np.ndarray, sampler_name: str | None = None):
        """
        Initialize the analyzer with samples and an optional sampler name.
        """
        self.delta = delta
        self.m = m
        self.samples = samples
        self.sampler_name = sampler_name

    def check_samples_are_no_signaling(
        self,
        save_path: str = None,
    ) -> bool:
        """
        Check that the samples are in the no-signaling set.
        """
        logger.trace("Checking that the points are in the no-signaling set")
        all_samples_good = True
        all_samples_count = 0
        bad_samples_count = 0
        for vec in tqdm(self.samples, desc="Checking samples"):
            all_samples_count += 1
            # Check that the point is in the no-signaling set
            behavior = RoutedBehavior(delta=self.delta, m=self.m, vector=vec)
            if not behavior.is_no_signaling():
                all_samples_good = False
                bad_samples_count += 1
                logger.error(f"Point is not in the no-signaling set:{behavior}")
        if all_samples_good:
            logger.success(f"All {all_samples_count} points are in the no-signaling set")
        else:
            logger.error(
                f"Some points are not in the no-signaling set ({bad_samples_count}/{all_samples_count})"  # noqa: E501
            )
        if save_path is not None:
            with open(save_path, "a") as f:
                f.write(f"Sampler: {self.sampler_name}\n")
                f.write(f"All samples are no-signaling: {all_samples_good}\n")
                f.write(f"Number of bad samples: {bad_samples_count}/{all_samples_count}\n")
        return all_samples_good

    def plot_projection(
        self,
        rng_seed: int = None,
        save_path: str = None,
        plot: bool = True,
        colors: list[bool] = None,
        alpha: float = 0.5,
    ):
        # Get the dimension of the samples
        d = self.samples.shape[1]

        # Pick a random behavior from the samples
        if rng_seed is not None:
            with np.random.seed(rng_seed):
                ref_vector_idx = np.random.choice(self.samples.shape[0])
                ref_vector = self.samples[ref_vector_idx]
                indices = np.random.choice(range(d), 2, replace=False)
        else:
            ref_vector_idx = np.random.choice(self.samples.shape[0])
            ref_vector = self.samples[ref_vector_idx]
            indices = np.random.choice(range(d), 2, replace=False)

        # Build a projector onto a plane orthogonal to the vector
        def projector(v):
            """
            Build a projector onto a plane orthogonal to the vector v
            """
            # Build the projector matrix
            P = np.eye(d) - np.outer(v, v) / np.dot(v, v)
            return P[indices, :]  # Keep only the first two rows to get a 2D projector

        proj = projector(ref_vector)

        # Project the samples onto the plane
        projected_samples = np.dot(proj, self.samples.T).T

        # Plot the samples

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot
        if colors is None:
            colors = ["blue"] * len(self.samples)
        else:
            colors = ["yellow" if c else "blue" for c in colors]
        axs[0].scatter(
            projected_samples[:, 0],
            projected_samples[:, 1],
            c=colors,
            marker="+",
            alpha=alpha,
        )
        axs[0].scatter(
            projected_samples[ref_vector_idx, 0],
            projected_samples[ref_vector_idx, 1],
            c="red",
            marker="x",
            s=100,
            label="Reference vector",
        )

        axs[0].set_title(f"Samples projected onto a plane orthogonal to {ref_vector_idx}-th vector")
        axs[0].set_xlabel("Projected dimension 1")
        axs[0].set_ylabel("Projected dimension 2")

        # 2D histogram (heatmap)
        h = axs[1].hist2d(
            projected_samples[:, 0], projected_samples[:, 1], bins=100, cmap="viridis"
        )
        plt.colorbar(h[3], ax=axs[1], label="Sample count")
        axs[1].set_xlabel("Projected dimension 1")
        axs[1].set_ylabel("Projected dimension 2")
        axs[1].set_title("Sampling density heatmap on projected plane")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
            logger.success(f"Saved plot to {save_path}")
        if plot:
            plt.show()
        return None

    def global_analyze_sampling(self, plot=False, log=True):
        """
        Analyze the uniformity of a set of sampled points using pairwise distances.

        Parameters
        ----------
        points : ndarray
            Array of shape (N, d) where each row is a sample in R^d.
        plot : bool
            Whether to plot a histogram of pairwise distances.

        Returns
        -------
        dict
            Summary statistics: mean, std, min, max of pairwise distances.
        """
        if self.samples.ndim != 2:
            raise ValueError("Input must be a 2D array of shape (n_samples, n_dimensions)")

        # Compute all pairwise distances
        dists = pdist(self.samples, metric="euclidean")

        # Stats
        stats = {
            "mean": np.mean(dists),
            "std": np.std(dists),
            "min": np.min(dists),
            "max": np.max(dists),
        }

        # Print stats
        if log:
            logger.info(f"Pairwise distance stats: {stats}")
            logger.info(f"Mean distance: {stats['mean']:.7f}")
            logger.info(f"Std distance: {stats['std']:.7f}")
            logger.info(f"Min distance: {stats['min']:.7f}")
            logger.info(f"Max distance: {stats['max']:.7f}")

        if plot:
            plt.hist(dists, bins=50, alpha=0.7, edgecolor="k")
            plt.title("Pairwise Distance Distribution")
            plt.xlabel("Euclidean distance")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()

        return stats

    # Check the uniformity of the samples
    def analyze_local_uniformity(self, k=None, plot=False, log=True, save_text=None, save_fig=None):
        """
        Check the local uniformity of a point cloud using k-nearest neighbor distances.

        Parameters
        ----------
        points : ndarray of shape (N, d)
            Sampled points in R^d.
        k : int or None
            Number of neighbors to consider (default: d+1).
        plot : bool
            Whether to show a histogram of neighbor distances.

        Returns
        -------
        dict
            Statistics on neighbor distances.
        """
        N, d = self.samples.shape
        if k is None:
            k = d + 1  # exclude the point itself

        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(self.samples)
        distances, _ = nbrs.kneighbors(self.samples)

        # Exclude the zero distance to the point itself
        knn_distances = distances[:, 1:]

        # Flatten for global stats
        all_knn_dists = knn_distances.flatten()

        stats = {
            "mean": np.mean(all_knn_dists),
            "std": np.std(all_knn_dists),
            "min": np.min(all_knn_dists),
            "max": np.max(all_knn_dists),
            "per_point_std_mean": np.mean(np.std(knn_distances, axis=1)),
        }

        if save_text is not None:
            with open(save_text, "a") as f:
                f.write(f"Sampler: {self.sampler_name}\n")
                f.write(f"Local uniformity stats (k={k}): {stats}\n")
                f.write(f"Mean k-NN distance: {stats['mean']:.7f}\n")
                f.write(f"Std k-NN distance: {stats['std']:.7f}\n")
                f.write(f"Min k-NN distance: {stats['min']:.7f}\n")
                f.write(f"Max k-NN distance: {stats['max']:.7f}\n")
                f.write(f"Mean per-point std: {stats['per_point_std_mean']:.7f}\n")
                f.write("\n")

        # Print stats
        if log:
            logger.info(f"Local uniformity stats (k={k}): {stats}")
            logger.info(f"Mean k-NN distance: {stats['mean']:.7f}")
            logger.info(f"Std k-NN distance: {stats['std']:.7f}")
            logger.info(f"Min k-NN distance: {stats['min']:.7f}")
            logger.info(f"Max k-NN distance: {stats['max']:.7f}")
            logger.info(f"Mean per-point std: {stats['per_point_std_mean']:.7f}")

        if plot:
            plt.hist(all_knn_dists, bins=50, alpha=0.7, edgecolor="k")
            plt.title(f"{k}-NN Distance Distribution")
            plt.xlabel("Distance")
            plt.ylabel("Frequency")
            plt.grid(True)
        if save_fig is not None:
            plt.savefig(save_fig)
            logger.success(f"Saved plot to {save_fig}")
        if plot:
            plt.show()

        return stats

    def search_outliers(self, k=5, threshold=2.0):
        """
        Search for outliers in the sample set based on k-NN distances.

        Parameters
        ----------
        k : int
            Number of neighbors to consider.
        threshold : float
            Distance threshold to classify a point as an outlier.

        Returns
        -------
        list
            Indices of outlier points.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(self.samples)
        distances, _ = nbrs.kneighbors(self.samples)

        # Exclude the zero distance to the point itself
        knn_distances = distances[:, 1:]

        # Calculate mean and std for each point's k-NN distances
        mean_dists = np.mean(knn_distances, axis=1)
        std_dists = np.std(knn_distances, axis=1)

        # Identify outliers based on the threshold
        outliers = np.where(
            (mean_dists > (mean_dists.mean() + threshold * std_dists))
            | (mean_dists < (mean_dists.mean() - threshold * std_dists))
        )[0]

        logger.info(f"Found {len(outliers)} outliers based on k-NN distances.")
        return outliers


if __name__ == "__main__":
    # Example usage

    # Set up the experiment parameters
    delta = 2
    m = 2
    n_samples = int(1e5)
    n_burn = int(1e5)

    # Configure the logger for console output clarity
    logger.remove()
    logger.add(
        sink=sys.stdout,
        format="<level>{level:<10} | {message}</>",
        level="DEBUG",
        colorize=True,
    )

    # Name the saved files
    save_projections = "projection.png"
    save_uniform_histogram = "uniformity_histogram.png"
    save_uniform_text = "stats_exp.txt"

    # Create the sampler and generate samples
    sampler = NoSignalingSampler(delta, m)
    samples = sampler.sample_multiple(
        number_of_samples=n_samples,
        number_to_burn=n_burn,
    )

    # Instantiate the analyzer with the generated samples
    analyzer = SamplesAnalyzer(delta, m, samples)

    # Do work with the samples !
    analyzer.plot_projection(save_path=save_projections, plot=False)
    analyzer.analyze_local_uniformity(
        save_fig=save_uniform_histogram, save_text=save_uniform_text, plot=True
    )  # Crashes on big samples by memory overload, be wary.
    analyzer.search_outliers(k=5, threshold=2.0)
