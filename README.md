# full-dimensional-srns-explorer

Code and programmatic methods surrounding a research project at ULB on routed Bell experiments and the characterization of short-range no-signaling (SRNS) behaviors. This repository implements methods for working in the full-dimensional space of behaviors.

---

## Overview

This project provides tools for:

- Defining and manipulating behavior sets (including no-signaling and short-range no-signaling sets)
- Enumerating and analyzing polytopes arising in quantum information theory
- Sampling, projecting, and analyzing behaviors in high-dimensional spaces
- Extracting and studying facets and vertices of relevant polytopes

The code is modular, with each file focusing on a specific aspect of the SRNS polytope analysis pipeline.

---

## Code

The main code is in the `code/` directory and includes:

- **behaviors.py**  
  Abstract and concrete classes for behaviors (e.g., `RoutedBehavior`, `LatentSRNSBehavior`), with utilities for vector/matrix conversions and no-signaling checks.

- **no_signaling_sets.py**  
  Defines abstract and concrete classes for sets of behaviors, including methods to generate the equations defining no-signaling and SRNS polytopes.

- **samplers.py**  
  Tools for sampling uniformly from the no-signaling set, analyzing sample distributions, and plotting projections.

- **solve_full_polytope.py**  
  Scripts for enumerating vertices of the latent SRNS set, projecting them, and computing the polytope’s H-representation.

- **boxworld_solve.py**  
  Specialized tools for analyzing the boxworld polytope, including brute-force enumeration of extremal behaviors and measurement strategies.

- **get_non_srns.py**  
  Utilities for extracting non-SRNS points from sampled data, useful for facet analysis.

- **utils/**  
  Helper modules for equation manipulation, documentation extraction, and other utilities.

---

## Data

The `output/` directory contains results and intermediate files, such as:

- Vertices and H-representations of polytopes (e.g., `latent_polytope_vertices.txt`, `measured_h_representation_boxworld_polytope.txt`)
- Sampled behaviors and analysis results

---

## Getting Started

1. **Install dependencies:**  
   Most scripts require `numpy`, `scipy`, `loguru`, `cdd`, `tqdm`, and optionally `polytopewalk` and `scikit-learn`.
   A `pyproject.toml` file is provided at root level for dependency management.

   **NB:** This project also requires access to the C-based `cdd` library, along with its python wrapper `pycddlib`. This should be externally installed and available on the running system.

2. **Run analyses:**  
   - Use `solve_full_polytope.py` to enumerate and analyze the SRNS polytope.
   - Use `samplers.py` to generate and analyze samples from the no-signaling set.
   - Use `boxworld_solve.py` for boxworld-specific polytope computations.

3. **Explore outputs:**  
   Results are written to the `output/` directory for further analysis.

---

## References

- See the associated internship report for references and detailed explanations of the project.
  - Readers not in possession of said report are, for the time being, likely not supposed to view this repository.

---

## License

See `LICENSE` file for details.

## Additional info

As of Aug 20, 2025, we try to complete certain computations using [PANDA: Parallel AdjaceNcy Decomposition Algorithm](http://comopt.ifi.uni-heidelberg.de/software/PANDA/) (see the associated [publication](http://comopt.ifi.uni-heidelberg.de/software/PANDA/#publication) ; the [archive repo we used](https://github.com/stefanloerwald/panda), and a personal [fork](https://github.com/LgBossu/panda) introduced to fix the source code on my personal setup (details on github, or by contacting me directly)).
The path to the PANDA built executable is stored as an environment variable in `paths.env`, as `PANDA="path/to/panda"`.
This is expected in `call_panda.py`.
