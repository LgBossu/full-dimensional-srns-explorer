from typing import List

import numpy as np
from loguru import logger


def quickload_file(exp_file: str) -> List[np.ndarray]:
    seen = set()  # To deduplicate points
    vertices_list = []  # To store the vertices
    # Load the vertices from the file
    logger.info(f"Loading vertices from {exp_file}...")

    with open(exp_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            # Convert the line to a numpy array and deduplicate
            if ";" in line:
                # If the line contains info, split it
                _, point_str = line.split(";")
                point = np.array([float(x) for x in point_str.strip().split(",")])
            else:
                # If the line does not contain info, just parse the point
                point = np.array([float(x) for x in line.strip().split(",")])
            # We try to deduplicate under the orbit of symmetries
            if point.tobytes() not in seen:
                seen.add(point.tobytes())
                vertices_list.append(point)

    logger.info(f"Loaded {len(vertices_list)} unique vertices from {exp_file}")

    return vertices_list


if __name__ == "__main__":
    vertices1 = quickload_file("pruned_vertices_222.txt")
    vertices2 = quickload_file("222_pruned_with_info.txt")

    set1 = set(map(tuple, vertices1))
    set2 = set(map(tuple, vertices2))

    logger.info(f"Both files coincide : {set1 == set2}")
    logger.info(f"Coinciding vertices : {len(set1 & set2)}")
    logger.info(f"Unique vertices in file 1 : {len(set1 - set2)}")
    logger.info(f"Unique vertices in file 2 : {len(set2 - set1)}")
