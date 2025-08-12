"""
A script to get the complete polytope representation of the SRNS
set, ultimately in the experiment space.
Performs vertex enumeration on the latent SRNS set's space,
projects the vertices onto the experiment space, and then
computes the hyperplane representation of the resulting polytope.
"""

# Import necessary libraries
from enum import Enum
from pathlib import Path
from time import time

import cdd
import no_signaling_sets
import numpy as np
from loguru import logger


class OutputTypes(Enum):
    """
    Enum to define the types of output files.
    """

    # Vertices of the latent no-signaling set
    LATENT_VERTICES = "latent_vertices"
    # Vertices of the measured SRNS set, projected onto the experiment space from
    # the latent set (long path coordinates are marginals of the latent behavior)
    MEASURED_VERTICES = "measured_vertices"
    # Inequalities of the measured SRNS set, in H-representation
    MEASURED_H_REPRESENTATION = "measured_h_representation"


files_headers = {
    OutputTypes.LATENT_VERTICES: "Latent SRNS Set Vertices",
    OutputTypes.MEASURED_VERTICES: "Measured SRNS Set Vertices",
    OutputTypes.MEASURED_H_REPRESENTATION: "Measured SRNS Set H-representation",
}


class OutputFormatter:
    """
    A class to handle solver output formatting and writing to files.

    This class provides methods to format and write solver outputs in both plain
    text and CSV formats.
    It supports different output types and manages file naming conventions,
    including optional file identifiers.

    Args:
        output_dir (Path, optional): The directory where output files will be
        saved. Defaults to Path("output/").
        files_id (str, optional): An optional identifier to append to output
        filenames. Defaults to "".

    Methods:
        _get_plain_text_path(output_type: OutputTypes) -> Path:
            Get the output file path for plain text format based on the output
            type and files_id.

        _get_csv_path(output_type: OutputTypes) -> list[Path]:
            Get the output file path(s) for CSV format based on the output type
            and files_id.
            For MEASURED_H_REPRESENTATION, returns separate paths for equalities
            and inequalities.

        write_plain_text(
            Write the content to the specified output file in plain text format.
            For MEASURED_H_REPRESENTATION, requires lin_set to distinguish
            equalities.

        write_csv(
            Write the content to the specified output file(s) in CSV format.
            For MEASURED_H_REPRESENTATION, splits content into equalities and
            inequalities using lin_set.
            CSV files include a header line that should be pruned upon reading.
    """

    def __init__(self, output_dir: Path = Path("output/"), files_id: str = "") -> None:
        self.output_dir = output_dir
        self.files_id = files_id

    def _get_plain_text_path(self, output_type: OutputTypes) -> Path:
        """
        Get the output file path based on the output type and files_id.
        """
        filename = f"{self.files_id}.txt" if self.files_id else f"{output_type.value}.txt"
        return self.output_dir / filename

    def _get_csv_path(self, output_type: OutputTypes) -> list[Path]:
        """
        Get the output file path for CSV format based on the output type and files_id.
        """
        if output_type == OutputTypes.MEASURED_H_REPRESENTATION:
            # For H-representation, we distinguish between equality and inequality
            eq_filename = (
                f"equality_{self.files_id}.csv"
                if self.files_id
                else f"{output_type.value}_equality.csv"
            )
            ineq_filename = (
                f"inequality_{self.files_id}.csv"
                if self.files_id
                else f"{output_type.value}_inequality.csv"
            )
            res = [
                self.output_dir / eq_filename,
                self.output_dir / ineq_filename,
            ]

        else:
            filename = (
                f"{output_type.value}_{self.files_id}.csv"
                if self.files_id
                else f"{output_type.value}.csv"
            )
            res = [self.output_dir / filename]

        return res

    def write_plain_text(
        self,
        output_type: OutputTypes,
        content: list | np.ndarray,
        lin_set: set[int] | None = None,
    ) -> None:
        """
        Write the content to the specified output file.
        """
        output_file_path = self._get_plain_text_path(output_type)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_type == OutputTypes.MEASURED_H_REPRESENTATION and lin_set is None:
            raise ValueError("For MEASURED_H_REPRESENTATION, lin_set must be provided.")
        elif output_type != OutputTypes.MEASURED_H_REPRESENTATION and lin_set is not None:
            logger.warning(
                "lin_set is provided but not needed for this output type. It will be ignored."
            )

        if isinstance(content, np.ndarray):
            content = list(content)

        with open(output_file_path, "w") as f:
            f.write(f"{files_headers[output_type]}\n\n")
            if output_type == OutputTypes.MEASURED_H_REPRESENTATION:
                f.write(f"Linear set: {lin_set}\n")
            f.write(f"Number of entries: {len(content)}\n")
            if len(content) > 0:
                f.write(f"Entry dimension: {len(content[0])}\n\n")

            for entry in content:
                f.write(" ".join(map(str, entry)) + "\n")

        return None

    def write_csv(
        self,
        output_type: OutputTypes,
        content: list | np.ndarray,
        lin_set: set[int] | None = None,
    ) -> None:
        """
        Write the content to the specified output file in CSV format.
        """
        output_file_path = self._get_csv_path(output_type)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_type != OutputTypes.MEASURED_H_REPRESENTATION and lin_set is not None:
            logger.warning(
                "lin_set is provided but not needed for this output type. It will be ignored."
            )

        if isinstance(content, list):
            content = np.array(content)

        if output_type == OutputTypes.MEASURED_H_REPRESENTATION:
            assert lin_set is not None, "lin_set must be provided for H-representation."

            eq_file, ineq_file = output_file_path
            with open(eq_file, "w") as eq_f, open(ineq_file, "w") as ineq_f:
                eq_f.write(f"{files_headers[output_type]} - Equalities\n")
                ineq_f.write(f"{files_headers[output_type]} - Inequalities\n")

                for i, expr in enumerate(content):
                    if i in lin_set:
                        eq_f.write(",".join(map(str, expr)) + "\n")
                    else:
                        ineq_f.write(",".join(map(str, expr)) + "\n")

        else:
            output_file = output_file_path[0]
            with open(output_file, "w") as f:
                f.write(f"{files_headers[output_type]}\n")

                for entry in content:
                    f.write(",".join(map(str, entry)) + "\n")


class RepTypes(Enum):
    """
    Enum to define the types of representations for cdd.Polyhedron.
    """

    GENERATOR = cdd.RepType.GENERATOR
    INEQUALITY = cdd.RepType.INEQUALITY


class PolytopeTypes(Enum):
    """
    Enum to define the types of polytopes.
    """

    LATENT = "latent"
    MEASURED = "measured"


vertex_outputs = {
    PolytopeTypes.LATENT: OutputTypes.LATENT_VERTICES,
    PolytopeTypes.MEASURED: OutputTypes.MEASURED_VERTICES,
}


class PolytopeWrapper:
    """
    A wrapper class for cdd.Polyhedron to handle polytope operations.

    This class provides a convenient interface for creating, manipulating,
    and querying polytopes using the pycddlib library. It supports both
    H-representation (inequalities) and V-representation (generators),
    and provides methods for projection, containment checks, and exporting
    polytope data.
    Args:
        source_array (np.ndarray): The array representing the polytope, either as
        inequalities or generators.
        rep_type (RepTypes): The representation type (INEQUALITY or GENERATOR).
        lin_set (set[int] | None): Indices of linear constraints (for INEQUALITY
        representation).
    Methods:
        get_generators() -> np.ndarray:
            Get the generators (vertices and rays) of the polytope as a numpy
            array.
        get_inequalities() -> tuple[np.ndarray, set[int]]:
            Get the inequalities (H-representation) of the polytope and its linear
            set.
        write_vertices(polytope_type: PolytopeTypes, output_dir: Path = Path
        ("output/"), file_id: str = "") -> None:
        write_inequalities(polytope_type: PolytopeTypes, output_dir: Path = Path
        ("output/"), file_id: str = "") -> None:
        contains(vector: np.ndarray, tol: float = 1e-10) -> bool:
            Check if a single vector belongs to the polytope within a given
            tolerance.
        contains_all(vectors: np.ndarray, tol: float = 1e-10) -> np.ndarray:
            Check if all vectors in a list belong to the polytope. Returns a
            boolean array.
        project_single(vector: np.ndarray, projection_matrix: np.ndarray) -> np.
        ndarray:
        project_vertices(projection_matrix: np.ndarray) -> np.ndarray:
            Project all vertices of the polytope onto a new space defined by the
            projection matrix.
        project_and_wrap(projection_matrix: np.ndarray) -> "PolytopeWrapper":
            Project the polytope's vertices and return a new PolytopeWrapper
            instance with the projected polytope.
        is_bounded() -> bool:
            Check if the polytope is bounded (i.e., has no rays).
    """

    def __init__(
        self,
        source_array: np.ndarray,
        rep_type: RepTypes,
        lin_set: set[int] | None,
    ) -> None:
        """
        Initialize the PolytopeWrapper with a source array and representation type.
        """
        self.source_array = source_array
        self.rep_type = rep_type
        self.lin_set = lin_set

        if rep_type == RepTypes.INEQUALITY:
            if lin_set is None:
                # If the representation is INEQUALITY, lin_set must be provided
                raise ValueError(
                    "For INEQUALITY representation, lin_set must be provided to label linear constraints."  # noqa: E501
                )
            elif not isinstance(lin_set, set):
                # If lin_set is provided, it must be a set of integers
                raise TypeError("lin_set must be a set of integers.")

        elif rep_type == RepTypes.GENERATOR and lin_set is not None:
            logger.warning(
                "lin_set is provided but not needed for GENERATOR representation. It will be ignored."  # noqa: E501
            )

        # Create the cdd.Matrix and cdd.Polyhedron
        self.cdd_matrix = cdd.Matrix(source_array, number_type="float")
        self.cdd_matrix.rep_type = rep_type.value
        if rep_type == RepTypes.INEQUALITY:
            # If the representation is INEQUALITY, set the linear set
            self.cdd_matrix.lin_set = self.lin_set

        self.polyhedron = cdd.Polyhedron(self.cdd_matrix)

    def get_generators(self) -> np.ndarray:
        """
        Get the generators of the polytope as a numpy array.
        """
        return np.array(self.polyhedron.get_generators(), dtype=float)

    def get_inequalities(self) -> tuple[np.ndarray, set[int]]:
        """
        Get the inequalities of the polytope as a np.ndarray with its linear set.
        """
        ineq_matrix = self.polyhedron.get_inequalities()
        lin_set = ineq_matrix.lin_set
        ineqs = np.array([ineq for ineq in ineq_matrix])

        return ineqs, lin_set

    def write_vertices(
        self,
        polytope_type: PolytopeTypes,
        output_dir: Path = Path("output/"),
        file_id: str = "",
    ) -> None:
        """
        Write the vertices of the polytope to a file.
        """
        output_formatter = OutputFormatter(
            output_dir=output_dir,
            files_id=file_id,
        )
        vertices = self.get_generators()
        output_type = vertex_outputs[polytope_type]
        output_formatter.write_plain_text(
            output_type,
            vertices,
            lin_set=self.lin_set,
        )

    def write_inequalities(
        self,
        polytope_type: PolytopeTypes,
        output_dir: Path = Path("output/"),
        file_id: str = "",
    ) -> None:
        """
        Write the inequalities of the polytope to a file.
        """
        output_formatter = OutputFormatter(
            output_dir=output_dir,
            files_id=file_id,
        )
        ineqs, lin_set = self.get_inequalities()

        # Check if the polytope type is implemented
        if polytope_type == PolytopeTypes.LATENT:
            raise NotImplementedError(
                "Writing inequalities for the latent polytope is not implemented yet (no practical use for now)."  # noqa: E501
            )

        output_type = OutputTypes.MEASURED_H_REPRESENTATION
        output_formatter.write_plain_text(
            output_type,
            ineqs,
            lin_set=lin_set,
        )
        output_formatter.write_csv(
            output_type,
            ineqs,
            lin_set=lin_set,
        )

    def contains(self, vector: np.ndarray, tol: float = 1e-10) -> bool:
        """
        Check if a vector belongs to the polytope.
        """
        belongs = True
        h_matrix, lin_set = self.get_inequalities()
        for i, inequality in enumerate(h_matrix):
            dot_result = np.dot(inequality[1:], vector) + inequality[0]

            if i in lin_set and not abs(dot_result) < tol:
                # If the inequality is linear and the dot product is not close to zero, it does not belong  # noqa: E501
                belongs = False
                break
            elif i not in lin_set and dot_result < -tol:
                # If the inequality is nonlinear and the dot product is negative, it does not belong
                belongs = False
                break
        return belongs

    def contains_all(self, vectors: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """
        Check if all vectors in a list belong to the polytope. Uses numpy for efficient computation.
        """
        # AI GENERATED CODE
        h_matrix, lin_set = self.get_inequalities()
        vectors = np.atleast_2d(vectors)

        # Compute dot products for all inequalities and all vectors
        dot_results = h_matrix[:, 1:] @ vectors.T + h_matrix[:, [0]]

        # For linear constraints: abs(dot) < tol
        lin_mask = np.array([i in lin_set for i in range(len(h_matrix))])
        linear_ok = (
            np.all(np.abs(dot_results[lin_mask, :]) < tol, axis=0)
            if np.any(lin_mask)
            else np.ones(vectors.shape[0], dtype=bool)
        )

        # For inequalities: dot >= -tol
        ineq_mask = ~lin_mask
        ineq_ok = (
            np.all(dot_results[ineq_mask, :] >= -tol, axis=0)
            if np.any(ineq_mask)
            else np.ones(vectors.shape[0], dtype=bool)
        )

        return linear_ok & ineq_ok

    def project_single(
        self,
        vector: np.ndarray,
        projection_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Project a single vector onto a new space defined by the projection matrix.
        It is not intended for actual use, but rather to demonstrate the
        projection logic that gets optimized for batch processing in
        `project_vertices`, using numpy for efficient computation.
        """
        logger.warning(
            "project_single is not intended for actual use. "
            "Use project_vertices for wrapped batch processing of polytope vertices."
        )
        # Separate the Y coordinate (tag) from the vector
        tag = vector[0]  # The tag (0 or 1) : 0 means a ray, 1 means a vertex
        vec = vector[1:]  # The actual vector to project

        projected_vec = projection_matrix @ vec  # shape (d_measured,)
        # For a single vector,
        # broadcasting ensures proper projection of the line vector

        # Rebuild the proper tagged vertex or ray
        projected_vector = np.concatenate(([tag], projected_vec))

        return projected_vector

    def project_vertices(self, projection_matrix: np.ndarray) -> np.ndarray:
        """
        Project the vertices of the polytope onto a new space defined by the
        projection matrix.
        Uses numpy for efficient batch computation.

        See `project_single` for the original projection logic.
        """
        vertices = self.get_generators()

        tags = vertices[:, 0]  # The first column contains the tags (0 or 1)
        # Tags distinguish between rays (0) and vertices (1), and are required for
        # reconstruction from vertices (cdd.Matrix.RepType.GENERATOR)
        vecs = vertices[:, 1:]

        # Transposing the projection matrix to match the shape of vecs avoids
        # abusive broadcasting and ensures proper matrix multiplication
        projected_vecs = vecs @ projection_matrix.T

        # Reconstruct the projected vertices with their tags
        projected_vertices = np.column_stack((tags, projected_vecs))
        return projected_vertices

    def project_and_wrap(
        self,
        projection_matrix: np.ndarray,
    ) -> "PolytopeWrapper":
        """
        Project the polytope's vertices onto a new space defined by the projection matrix,
        and return a new PolytopeWrapper instance with the projected vertices.
        """
        projected_vertices = self.project_vertices(projection_matrix)
        projected_polytope = PolytopeWrapper(
            source_array=projected_vertices,
            rep_type=RepTypes.GENERATOR,
            lin_set=None,  # No linear set for a polytope from vertices
        )
        return projected_polytope

    def is_bounded(self) -> bool:
        """
        Check if the polytope is bounded.
        A polytope is bounded if it has no rays (i.e., all vertices are finite).
        """
        generators = self.get_generators()
        # Check if there are any rays (tag == 0)
        return not np.any(generators[:, 0] == 0.0)  # Tag == 0 indicates a ray


class PolytopeSolver:
    """
    A class to handle the polytope solving process.
    It wraps the cdd.Polyhedron and provides methods for vertex enumeration,
    projection, and representation writing.

    This class wraps the cdd.Polyhedron and provides methods for vertex
    enumeration,
    projection, and representation writing for polytopes arising from no-signaling
    sets.
    Args:
        delta (int): The number of settings or outcomes parameterizing the latent
        set.
        m (int): The number of measurements or parties parameterizing the latent
        set.
        output_dir (Path, optional): Directory where output files will be written.
        Defaults to Path("output/").
        solver_id (str, optional): Optional identifier for the solver instance.
        Defaults to "".
    Attributes:
        delta (int): The number of settings or outcomes.
        m (int): The number of measurements or parties.
        latent_set (LatentSRNSSet): The latent no-signaling set object.
        projection_matrix (np.ndarray): Matrix used to project latent space to
        measured space.
        latent_polytope (PolytopeWrapper | None): The polytope in the latent space.
        measured_polytope (PolytopeWrapper | None): The polytope in the measured
        (experiment) space.
    Methods:
        compute_latent_polytope():
        compute_measured_polytope():
        write_experiment_data():
            Write the latent and measured polytope data (vertices and inequalities)
            to files in the output directory.
        solve():
            Solve the polytope problem by computing the latent and measured
            polytopes,
            writing the results to files, and logging the process.

    """

    def __init__(
        self,
        delta: int,
        m: int,
        output_dir: Path = Path("output/"),
        solver_id: str = "",
    ) -> None:
        """
        Initialize the PolytopeSolver with delta and m parameters.
        """
        self.delta = delta
        self.m = m
        self.latent_set = no_signaling_sets.LatentSRNSSet(delta, m)
        self.projection_matrix: np.ndarray = self.latent_set.latent_to_measured()

        self.latent_polytope: PolytopeWrapper | None = None
        self.measured_polytope: PolytopeWrapper | None = None

    def compute_latent_polytope(self) -> None:
        """
        Compute the latent polytope from the latent SRNS set.
        """
        latent_matrix_form, lin_list = self.latent_set.get_cdd_matrix()

        self.latent_polytope = PolytopeWrapper(
            source_array=latent_matrix_form,
            rep_type=RepTypes.INEQUALITY,
            lin_set=set(lin_list),  # Label the equations as such, using the linear set
        )

    def compute_measured_polytope(self) -> None:
        """
        Compute the measured polytope by projecting the latent polytope's vertices
        onto the experiment space.
        """
        if self.latent_polytope is None:
            raise ValueError(
                "Latent polytope must be computed before measured polytope. Please call compute_latent_polytope() first."  # noqa: E501
            )

        self.measured_polytope = self.latent_polytope.project_and_wrap(
            projection_matrix=self.projection_matrix
        )

    def write_experiment_data(self) -> None:
        """
        Write the latent and measured polytope data to files.
        """
        if self.latent_polytope is None:
            raise ValueError("Latent polytope must be computed before writing data.")

        # Write latent polytope vertices
        self.latent_polytope.write_vertices(
            polytope_type=PolytopeTypes.LATENT,
            output_dir=Path("output/"),
            file_id=f"delta_{self.delta}_m_{self.m}",
        )

        # Write measured polytope vertices
        if self.measured_polytope is None:
            raise ValueError("Measured polytope must be computed before writing data.")
        self.measured_polytope.write_vertices(
            polytope_type=PolytopeTypes.MEASURED,
            output_dir=Path("output/"),
            file_id=f"delta_{self.delta}_m_{self.m}",
        )

        # Write measured polytope inequalities (H-representation)
        self.measured_polytope.write_inequalities(
            polytope_type=PolytopeTypes.MEASURED,
            output_dir=Path("output/"),
            file_id=f"delta_{self.delta}_m_{self.m}",
        )

    def solve(self) -> None:
        """
        Solve the polytope by computing the latent and measured polytopes,
        and writing the results to files.
        """
        logger.info("Computing latent polytope...")
        start_time = time()
        self.compute_latent_polytope()
        logger.info(f"Latent polytope computed ({time() - start_time:.2f}s).")

        logger.info("Computing measured polytope...")
        start_time = time()
        self.compute_measured_polytope()
        logger.info(f"Measured polytope computed ({time() - start_time:.2f}s).")

        self.write_experiment_data()

        assert (
            self.latent_polytope is not None
        ), "Latent polytope is not initialized. This error should not happen at this point."  # noqa: E501
        assert (
            self.measured_polytope is not None
        ), "Measured polytope is not initialized. This error should not happen at this point."  # noqa: E501

        logger.info("Writing experiment data to files.")
        logger.success(
            f"Experiment data written. Computed:\n    {self.latent_polytope.get_generators().shape[0]} latent vertices,\n    {self.measured_polytope.get_generators().shape[0]} measured vertices,\n    {self.measured_polytope.get_inequalities()[0].shape[0]} inequalities."  # noqa: E501
        )
        logger.success("Terminated successfully.")
        return None


if __name__ == "__main__":
    # Dimensions and parameters
    delta = 2
    m = 2
    limit_testing = None

    # Initialize the solver
    solver = PolytopeSolver(delta=delta, m=m)
    # Solve the polytope
    solver.solve()

    # Compare the measured polytope with sampled data
    data_dir = Path("data/view_srns/")
    experiment_suffix = "_delta_2_m_2_samples_1000000_runtime_1747837663.9102557.npy"

    data_points = np.load(data_dir / f"sampled_behaviors{experiment_suffix}")
    belonging = np.load(data_dir / f"belonging_list{experiment_suffix}")
    belonging = np.array(belonging[:, 1]).astype(bool)  # Convert to boolean

    # Get only the limited number of data points specified
    data_points = data_points[:limit_testing]
    belonging = belonging[:limit_testing]

    assert (
        solver.measured_polytope is not None
    ), "Measured polytope is not initialized. This error should not happen at this point."  # noqa: E501
    inferred_belonging = solver.measured_polytope.contains_all(data_points)

    # Dissimilarity check
    dissimilarity = np.mean((inferred_belonging != belonging)) * 100

    if dissimilarity != 0:
        logger.error(
            "The measured polytope does not match the sampled data within the threshold. "
            f"Dissimilarity: {dissimilarity:.2f}%."
        )
    else:
        logger.success("The measured polytope matches the sampled data perfectly.")

    # Boundedness check
    if solver.measured_polytope.is_bounded():
        logger.success("The measured polytope is bounded.")
    else:
        logger.warning("The measured polytope is unbounded.")
