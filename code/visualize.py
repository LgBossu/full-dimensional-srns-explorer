import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html
from loguru import logger


class DataLoader:
    def __init__(
        self,
        delta: int,
        m: int,
        size_limit: int = int(1e7),
        data_dir: str | Path = Path("data/view_srns"),
    ):
        self.delta = delta
        self.m = m
        self.size_limit = size_limit
        self.data_dir = Path(data_dir)

    def parse_file_id(
        self,
        filename: str,
    ) -> tuple[int, int, int, int]:
        """
        Parse the file name to extract the delta and m values.
        Works based on the default naming conventions of the data directory.
        Requires linux file structure.
        """
        filename = filename.split("/")[-1]
        parts = filename.split(".")[0].split("_")

        try:
            delta_idx = parts.index("delta")
            m_idx = parts.index("m")
            samples_idx = parts.index("samples")
            runtime_idx = parts.index("runtime")
        except ValueError:
            raise ValueError(f"Filename {filename} does not follow the expected naming convention.")

        delta = int(parts[delta_idx + 1])
        m = int(parts[m_idx + 1])
        samples = int(parts[samples_idx + 1])
        file_id = int(parts[runtime_idx + 1])
        return delta, m, samples, file_id

    def count_matching_files(
        self,
        filenames_list: list[str],
    ) -> dict[int, int]:
        counts = dict()
        for filename in filenames_list:
            _, _, _, file_id = self.parse_file_id(filename)
            if file_id not in counts:
                counts[file_id] = 1
            else:
                counts[file_id] += 1
        return counts

    def determine_best_files(
        self,
        require_tagged: bool = True,
    ) -> list[str]:
        """
        Determine the best files to load based on the size limit
        and experiment dimensions imposed.
        Works based on the default naming structure of the data directory.
        """
        all_files = os.listdir(self.data_dir)
        all_files = [f for f in all_files if f.endswith(".npy")]
        matching_files = [
            f
            for f in all_files
            if self.parse_file_id(f)[0] == self.delta and self.parse_file_id(f)[1] == self.m
        ]

        if not require_tagged:
            tagged_files = matching_files
        else:
            # require_tagged : i.e, we want datasets with a matching belongings list
            counts = self.count_matching_files(all_files)
            tagged_files = [
                filename for filename in all_files if counts[self.parse_file_id(filename)[3]] > 1
            ]

        # Filter files based on size limit
        sizes = [self.parse_file_id(filename)[2] for filename in tagged_files]

        # Sort sizes and tagged_files based on sizes in descending order
        sorted_pairs = sorted(
            zip(sizes, tagged_files),
            key=lambda x: x[0],
            reverse=True,
        )
        sizes, tagged_files = zip(*sorted_pairs) if sorted_pairs else ([], [])

        # Filter files based on size limit
        beyond_limit = 0
        for i, size in enumerate(sizes):
            if size > self.size_limit:
                beyond_limit += 1
            else:
                break

        # If all files are beyond the limit, raise an error
        if beyond_limit == len(sizes):
            raise ValueError(
                f"All files are beyond the size limit of {self.size_limit} bytes. Please increase the size limit or sample smaller files."  # noqa: E501
            )

        # Keep only files within the size limit
        tagged_files = tagged_files[beyond_limit:]
        sizes = sizes[beyond_limit:]

        files_to_use = [0]
        tot_size = sizes[0]
        cur = 1

        while cur < len(sizes) and tot_size < self.size_limit:
            if sizes[cur] + tot_size <= self.size_limit:
                files_to_use.append(cur)
                tot_size += sizes[cur]
            cur += 1

        files_to_use = [tagged_files[i] for i in files_to_use]

        return files_to_use

    def load_data(
        self,
        filenames: list[str],
        tagged: bool = True,
    ) -> list[np.ndarray]:
        loaded_arrays = []
        if tagged:
            tags_list = []

        for filename in filenames:
            data = np.load(self.data_dir / filename)
            loaded_arrays.append(data)

            if tagged:
                tag_array_name = filename.split("/")[-1]
                tag_array_name = tag_array_name.replace("sampled_behaviors", "belonging_list")
                tag_array = np.load(self.data_dir / tag_array_name)
                tags_list.append([bool(x[1]) for x in tag_array])

        # Concatenate all loaded arrays along the first axis
        data = np.vstack(loaded_arrays)
        dataset = [data]

        if tagged:
            tags = np.array(tags_list)
            dataset = [data[tags], data[~tags]]

        return dataset

    def autoload(
        self,
        require_tagged: bool = True,
    ) -> list[np.ndarray]:
        """
        Automatically load the data based on the delta and m values.
        """
        filenames = self.determine_best_files(
            require_tagged=require_tagged,
        )
        return self.load_data(filenames, tagged=require_tagged)


class DataSlicer:
    def __init__(
        self,
        delta: int,
        m: int,
        data: list[np.ndarray],
        slice_relative_thickness: float = 0.01,
        axes: tuple[int, int] | None = None,
    ):
        # Primitive arguments
        self.data = data
        self.delta = delta
        self.m = m

        # Estimate typical size of the data
        minimum = np.inf
        maximum = -np.inf
        for data_array in data:
            mini = np.min(data_array)
            maxi = np.max(data_array)
            if mini < minimum:
                minimum = mini
            if maxi > maximum:
                maximum = maxi
        self.min = minimum
        self.max = maximum

        # Slicing parameters
        self.epsilon: float = slice_relative_thickness * (self.max - self.min)
        self.n_slices: int = ((self.max) - (self.min) // self.epsilon) + 1

        self.slice_values = np.linspace(
            self.min,
            self.max,
            num=self.n_slices,
            endpoint=True,
        )
        logger.debug(f"Computed slice values: {self.slice_values}")
        logger.debug(f"Computed epsilon: {self.epsilon}")
        logger.debug(f"Computed of slices: {self.n_slices}")

        # Slicing projection and direction
        if axes is None:
            axes = (0, self.delta**2 + self.m**2)
        self.x_axis = axes[0]
        self.y_axis = axes[1]
        self.slice_direction = np.ones(2 * self.delta**2 + 2 * self.m**2)
        self.slice_direction[self.x_axis] = 0
        self.slice_direction[self.y_axis] = 0

        # Slicing knife
        self.slice_distance = self.slice_direction * self.epsilon
        self.slice_distance[self.x_axis] = np.inf
        self.slice_distance[self.y_axis] = np.inf

    def slice_data(
        self,
        data: list[np.ndarray],
    ) -> list[list[np.ndarray]]:
        """
        Slice the data based on the given epsilon and axes.
        """
        # AI GENERATED CODE
        bounds = np.stack(
            [
                self.slice_values * self.slice_direction - self.slice_distance,
                self.slice_values * self.slice_direction + self.slice_distance,
            ]
        )

        out = []
        for arr in data:
            out.append([arr[((low < arr) & (arr <= up)).all(axis=1)] for low, up in bounds.T])

        return out

    def get_app(self) -> Dash:
        """
        Get the Dash app for visualization.
        """
        app = Dash(__name__)

        # Define the layout of the app
        app.layout = html.Div(
            [
                dcc.Graph(id="graph"),
                dcc.Slider(
                    id="slice-slider",
                    min=0,
                    max=self.n_slices - 1,
                    value=0,
                    marks={i: str(i) for i in range(self.n_slices)},
                ),
            ]
        )

        # Define the callback to update the graph based on the slider value
        @callback(
            Output("graph", "figure"),
            Input("slice-slider", "value"),
        )
        def update_graph(slice_index):
            fig = go.Figure()
            for i, data_array in enumerate(self.data):
                fig.add_trace(
                    go.Scatter(
                        x=data_array[:, self.x_axis],
                        y=data_array[:, self.y_axis],
                        mode="markers",
                        name=f"Points {i}",
                    )
                )
            return fig

        return app
