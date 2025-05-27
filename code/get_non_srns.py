"""
A little utility script defining functions to help identify and
extract non-srns points from a sampled distribution of points in
NS.

Non-SRNS points are of particular interest, since performing
linear programming on them to test for belonging in SRNS yields,
in dual variables, equations to potential SRNS facets. We thus
isolate them in aggregated files for further analysis.
"""

import os
from pathlib import Path

import numpy as np
from loguru import logger

# Define directories
non_srns = Path("data/non_srns")
points_data = Path("data/view_srns")


# Parse data dir
def parse_data_dir(dir_list: list[str]) -> list[tuple[Path, Path]]:
    """
    Parse the data directory to get the list of files.
    Gets as an argument the listdir of the data directory.
    Returns a list of tuples (samples_file_path, belonging_list_file_path).
    """
    parsed_files = dict()
    res = []

    for file in dir_list:
        if file.endswith(".npy"):
            # We target numpy files to get the samples and corresponding data
            file_id = file.split("_")[-1].split(".")[0]
            if file_id in parsed_files:
                if file.split("/")[-1].startswith("sampled_behaviors"):
                    res.append((Path(file), Path(parsed_files[file_id])))
                elif file.split("/")[-1].startswith("belonging_list"):
                    res.append((Path(parsed_files[file_id]), Path(file)))
            else:
                parsed_files[file_id] = file
        else:
            # Not interested in figures, and others
            continue

    return res


# Extract from a sample of points the non-srns ones
def extract_non_srns(points_file: Path, belonging_list_file: Path) -> list[np.ndarray]:
    """
    Extract the non-srns points from a sample of points.
    """
    # Load the data (may be RAM intensive on large sample files)
    points = np.load(points_data / points_file)
    belonging_list = np.load(points_data / belonging_list_file)

    logger.debug(f"Points file content looks like: {points.shape}")

    # Order is preserved so the first coordinate of the belonging list (point rank) is useless
    belonging_list = [bool(x[1]) for x in belonging_list]

    # Get the non-srns points
    non_srns_points = points[np.logical_not(belonging_list)]

    logger.debug(f"Non-srns points file content looks like: {non_srns_points.shape}")

    return list(non_srns_points)
    # Discard srns points -- they are not needed for certain tasks


if __name__ == "__main__":
    # Usage

    # Get the list of files in the data directory
    try:
        dir_list = os.listdir(points_data)
    except FileNotFoundError as e:
        print(
            "Must run from the root directory of the repository."
        )  # We rely on relative paths for privacy
        raise e

    # Parse the data directory to get the list of files
    parsed_files = parse_data_dir(dir_list)

    # Loop over the files and extract the non-srns points
    non_srns_points = []
    for points_file, belonging_list_file in parsed_files:
        non_srns_points.extend(extract_non_srns(points_file, belonging_list_file))

    logger.debug(f"Non-srns points file content looks like: {len(non_srns_points)}")

    np.save(non_srns / "non_srns_points.npy", non_srns_points)
