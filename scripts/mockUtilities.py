## Some helper functions

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
import csv

from matplotlib.patches import Rectangle, Circle, RegularPolygon


## Class for various shapes
class RecObj:
    def __init__(self, anchor, width, height, angle):
        self._anchor = anchor
        self._width = width
        self._height = height
        self._angle = angle
        self._rectangle = Rectangle(
            self._anchor, self._width, self._height, angle=self._angle
        )

    def plot(self, ax, color):
        rectangle = Rectangle(
            self._anchor,
            self._width,
            self._height,
            angle=self._angle,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(rectangle)

    def contains(self, point):
        return self._rectangle.contains_point(point)

    def gradient_multiplier(self, point):
        # Returns a multiplier indicating a multiplier indicating the position along the rectangle of the point value from 0 to 1
        # NB: only works for vertical upright
        if self._angle != 0:
            raise ValueError(
                "Gradient method only works for rectangles that are upright (angle parameter is 0); this rectangle is not upright"
            )
        if not self.contains(point):
            raise ValueError("Point is not inside the rectangle")
        y = point[1]
        anchor_y = self._anchor[1]
        return (y - anchor_y) / self._height


class CircObj:
    def __init__(self, anchor, radius):
        self._anchor = anchor
        self._radius = radius
        self._circle = Circle(self._anchor, self._radius)

    def plot(self, ax, color):
        circle = Circle(
            self._anchor,
            self._radius,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(circle)

    def contains(self, point):
        return self._circle.contains_point(point)


class RegPolyObj:
    def __init__(self, anchor, radius, n_vertices, angle):
        self._anchor = anchor
        self._radius = radius
        self._n_vertices = n_vertices
        self._angle_rad = angle / 180 * np.pi
        self._polygon = RegularPolygon(
            self._anchor,
            radius=self._radius,
            numVertices=self._n_vertices,
            orientation=self._angle_rad,
        )

    def plot(self, ax, color):
        polygon = RegularPolygon(
            self._anchor,
            radius=self._radius,
            numVertices=self._n_vertices,
            orientation=self._angle_rad,
            edgecolor="black",
            facecolor=color,
        )
        ax.add_patch(polygon)

    def contains(self, point):
        return self._polygon.contains_point(point)


def rectangles_from_param_list(params):
    # Expects params to be a iterable with elements being tuples of the format: (({anchor_x, anchor_y}), {width}, {height}, {angle})
    return [RecObj(*param) for param in params]


shape_dict = {
    "rect": RecObj,
    "regPoly": RegPolyObj,
    "circ": CircObj,
}


def shapes_from_param_list(params):
    # Expects params to be a iterable with elements being tuples of format ({shape_type}, {tuple containing parameters for the shape type})
    return [shape_dict[shape_type](*param) for shape_type, param in params]


def plot_rectangles(rectangles, min_x, max_x, min_y, max_y, x_label=None, y_label=None, x_ticks=None, y_ticks=None, title=None):
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(range(0, 256, int(256 / len(rectangles))))
    for i, rec in enumerate(rectangles):
        rec.plot(ax, colors[i])
    ax.set_xlim(min_x-0.5, max_x-0.5)
    ax.set_ylim(min_y-0.5, max_y-0.5)
    if title:
        ax.set_title(title, fontsize=24)
    if x_label:
        ax.set_xlabel(x_label, fontsize=24)
    if y_label:
        ax.set_ylabel(y_label, fontsize=24)
    if x_ticks is not None:
        ax.set_xticks(x_ticks, fontsize=24)
    if y_ticks is not None:
        ax.set_yticks(y_ticks, fontsize=24)
    ax.set_aspect("equal")
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.show()


plot_shapes = plot_rectangles

## Mock Data Generation code


def get_gene_panel_df(n_genes):
    data = pd.DataFrame(
        {"x": [f"G{i}" for i in range(1, n_genes + 1)]},
        index=[str(i) for i in range(1, n_genes + 1)],
    )
    return data


def generate_gene_panel(filename, n_genes):
    data = get_gene_panel_df(n_genes)
    data.to_csv(filename, quoting=csv.QUOTE_NONNUMERIC)


def create_factor_gene_matrix(factor_active_genes, n_factors, n_genes):
    matrix = np.zeros((n_genes, n_factors), dtype=float)
    for factor, active_genes in enumerate(factor_active_genes):
        for gene in active_genes:
            matrix[gene, factor] = (
                1  # set the factor loading on the gene to 1, in future might want to do some kind of normalization (perhaps on a factor basis or gene basis), really depends on how wanna set it up, but no normalization is done for now
            )
    return matrix


def get_spot_factors(spot_coordinates, shapes):
    # Expect spot coordinates to be a tuple of form ({x}, {y})
    relevant_factors = []
    for i, rec in enumerate(shapes):
        if rec.contains(spot_coordinates):
            relevant_factors.append(i)
    return relevant_factors


def create_factor_spot_matrix(shapes, n_factors, n_x_indices, n_y_indices):
    # Presumes spots are indexed in a row by row form
    matrix = np.zeros((n_factors, n_y_indices * n_x_indices), dtype=float)
    row_len = n_y_indices
    for row in range(n_y_indices):
        for col in range(n_x_indices):
            spot_coordinates = (col, row)  # col is x, row is y
            relevant_factors = get_spot_factors(spot_coordinates, shapes)
            for factor in relevant_factors:
                matrix[factor, row * row_len + col] = (
                    1  # sets factor value to 1; now arguably this might need adjustment, e.g. perhaps normalized on a spot basis (so each spot's factor loading sums to 1) or factor basis, really depends on how wanna set it up, but not normalised for now
                )
    return matrix


def get_spot_factors_gradient(spot_coordinates, shapes):
    # Expect spot coordinates to be a tuple of form ({x}, {y})
    relevant_factors = []
    relevant_factor_multipliers = []
    for i, rec in enumerate(shapes):
        if rec.contains(spot_coordinates):
            relevant_factors.append(i)
            relevant_factor_multipliers.append(
                rec.gradient_multiplier(spot_coordinates)
            )
    return (relevant_factors, relevant_factor_multipliers)


def create_factor_spot_matrix_gradient(shapes, n_factors, n_x_indices, n_y_indices):
    # Presumes spots are indexed in a row by row form
    matrix = np.zeros((n_factors, n_y_indices * n_x_indices), dtype=float)
    row_len = n_y_indices
    for row in range(n_y_indices):
        for col in range(n_x_indices):
            spot_coordinates = (col, row)  # col is x, row is y
            relevant_factors, multiplier_values = get_spot_factors_gradient(
                spot_coordinates, shapes
            )
            for factor, multiplier in zip(relevant_factors, multiplier_values):
                matrix[factor, row * row_len + col] = (
                    multiplier  # sets factor value to 1; now arguably this might need adjustment, e.g. perhaps normalized on a spot basis (so each spot's factor loading sums to 1) or factor basis, really depends on how wanna set it up, but not normalised for now
                )
    return matrix


def spot_feature_array_to_matrix(spot_array, n_x_indices, n_y_indices):
    n_rows = n_y_indices
    n_cols = n_x_indices
    return np.reshape(spot_array, (n_rows, n_cols), order="C")


def spot_feature_matrix_to_dataframe(spot_matrix, n_x_indices, n_y_indices):
    # Prepare the dataframe
    x_indices = list(range(n_x_indices))
    y_indices = list(range(n_y_indices))
    multi_index = pd.MultiIndex.from_product(
        [y_indices, x_indices], names=["y_index", "x_index"]
    )
    n_features = spot_matrix.shape[0]
    colnames = [f"feature_{i}" for i in range(n_features)]

    df_data = spot_matrix.transpose()
    df = pd.DataFrame(df_data, index=multi_index, columns=colnames)
    return df


def plot_matrix(matrix, fig_size=(5, 4), cmap="viridis", x_label="x", y_label="y", x_ticks=None, y_ticks=None):
    # Set the width and height of the figure
    fig, ax = plt.subplots(figsize=fig_size)
    # Create a heatmap plot
    im = ax.imshow(
        matrix, cmap=cmap, interpolation="none", aspect="auto", origin="lower"
    )
    # Add colorbar for reference
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    # Set axis labels
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    if x_ticks is not None:
        ax.set_xticks(x_ticks, fontsize=24)
    if y_ticks is not None:
        ax.set_yticks(y_ticks, fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=16)
    # Show the plot
    plt.show()


def plot_pandas_multi_index_matrix_series(
    series, fig_size=(5, 4), cmap="viridis", x_label="x", y_label="y"
):
    # Expects the series to have multi index being x_index and y_index
    fig, ax = plt.subplots(figsize=fig_size)
    heatmap_data = series.unstack()
    # Create a heatmap plot
    im = ax.imshow(
        heatmap_data, cmap=cmap, interpolation="none", aspect="auto", origin="lower"
    )
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    # Set axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Show the plot
    plt.show()


## Testing functions
assert np.array_equal(
    spot_feature_array_to_matrix([1, 2, 3, 4, 5, 6], 3, 2),
    np.array([[1, 2, 3], [4, 5, 6]]),
)
