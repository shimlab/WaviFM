import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from .utilities import *
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

# Some visualisation helper functions


def plot_sparsity(sparsity, x_label, y_label, figsize=(5, 5)):
    # Expects a 2D numpy array with array values as the sparsity value (should be between 0 and 1)
    # Such array could be, e.g. the eta array for sparsity of F, or the flattened pi array for sparsity of L (flattened is so that the pi which is 4D in our setup is reduced to 2D by flattening the ijk indices (wavelet coefficient indices) to a single 1D index set)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    cax = ax.imshow(
        sparsity, cmap="viridis", vmin=0, vmax=1, interpolation="none", aspect="auto"
    )
    ax.invert_yaxis()
    fig.colorbar(cax)  # Add color bar indicating the scale
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks([0, sparsity.shape[1] - 1], [0, sparsity.shape[1] - 1])
    plt.yticks([0, sparsity.shape[0] - 1], [0, sparsity.shape[0] - 1])

    plt.show()


def plot_sparsity_thresholded(
    sparsity, x_label, y_label, figsize=(5, 5), upper_threshold=0.6, lower_threshold=0.4
):
    # Expects a 2D numpy array with array values as the sparsity value (should be between 0 and 1)
    # Values above upper threshold rounded to 1 for plotting sake, those below lower_threshold rounded to 0 for plotting sake
    # Such array could be, e.g. the eta array for sparsity of F, or the flattened pi array for sparsity of L (flattened is so that the pi which is 4D in our setup is reduced to 2D by flattening the ijk indices (wavelet coefficient indices) to a single 1D index set)
    rounded = threshold_to_0_or_1(
        sparsity, upper_threshold=upper_threshold, lower_threshold=lower_threshold
    )
    plot_sparsity(rounded, x_label, y_label, figsize=figsize)


def plot_list_sparsity(
    sparsity_list,
    x_label,
    y_label,
    figsize=(5, 5),
    suptitle=None,
):
    n_plots = len(sparsity_list)
    fig, axes = plt.subplots(n_plots, figsize=figsize)
    for i, sparsity in zip(range(n_plots), sparsity_list):
        ax = axes[i]
        im = ax.imshow(
            sparsity,
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="none",
            aspect="auto",
        )
        ax.invert_yaxis()
        im.set_clim(0, 1)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xticks([0, sparsity.shape[1] - 1], [0, sparsity.shape[1] - 1])
        ax.set_yticks([0, sparsity.shape[0] - 1], [0, sparsity.shape[0] - 1])

    # Set overall title

    if suptitle:
        fig.suptitle(suptitle)

    # Set colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def plot_list_sparsity_thresholded(
    sparsity_list,
    x_label,
    y_label,
    figsize=(5, 5),
    suptitle=None,
    upper_threshold=0.6,
    lower_threshold=0.4,
):
    rounded_sparsity_list = [
        threshold_to_0_or_1(
            sparsity, upper_threshold=upper_threshold, lower_threshold=lower_threshold
        )
        for sparsity in sparsity_list
    ]
    plot_list_sparsity(
        rounded_sparsity_list,
        x_label,
        y_label,
        figsize,
        suptitle,
    )


# Plotting utilities to help comparison of sparsity results
def compare_sparsity_lists(
    sparsity_list_1,
    sparsity_list_2,
    x_label,
    y_label,
    figsize=(5, 5),
    col_titles=None,
    suptitle=None,
):
    # Expects {sparsity_list_1} and {sparsity_list_2} to have the same dimensions
    assert len(sparsity_list_1) == len(sparsity_list_2)
    n_plots = len(sparsity_list_1)
    fig, axes = plt.subplots(n_plots, 2, figsize=figsize)
    for i in range(n_plots):
        for j, sparsity in enumerate((sparsity_list_1[i], sparsity_list_2[i])):
            if n_plots == 1:
                ax = axes[j]
            else:
                ax = axes[i][j]
            im = ax.imshow(
                sparsity,
                cmap="viridis",
                vmin=0,
                vmax=1,
                interpolation="none",
                aspect="auto",
            )
            ax.invert_yaxis()
            im.set_clim(0, 1)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_xticks([0, sparsity.shape[1] - 1], [0, sparsity.shape[1] - 1])
            ax.set_yticks([0, sparsity.shape[0] - 1], [0, sparsity.shape[0] - 1])

    # Set column and row titles
    if col_titles:
        for j in range(2):
            if n_plots == 1:
                axes[j].set_title(col_titles[j])
            else:
                axes[0, j].set_title(col_titles[j])

    # Set overall title

    if suptitle:
        fig.suptitle(suptitle)

    # Set colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def plt_elbo_record(elbo_record, marker="o", linestyle="solid", markersize=3):
    plt.plot(elbo_record, marker=marker, linestyle=linestyle, markersize=markersize)
    plt.xlabel("Iteration")
    plt.ylabel("ELBO")
    plt.title("ELBO")
    plt.show()


def create_custom_colormap(
    lower_color="blue", center_color="white", higher_color="red"
):
    """
    Creates a custom colormap that maps 0 to center_color and interpolates values
    above and below 0 in gradients towards lower_color and higher_color respectively.

    Parameters:
    - lower_color: The color for values less than 0.
    - center_color: The color for the value 0.
    - higher_color: The color for values greater than 0.

    Returns:
    - A LinearSegmentedColormap object.
    """
    # Define the color transitions
    colors = [lower_color, center_color, higher_color]
    # Create a colormap with three segments: below 0, at 0, and above 0
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    return cmap


def plot_matrices_side_by_side(
    matrix1,
    matrix2,
    matrix3,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False,
    fig=None,
    axes=None,
    aspect=None,
    x_ticks=None,
    y_ticks=None,
    number_size=16,
    figsize=(15,5),
    exact_color_ticks=False,
):
    """
    Plots three 2D matrices side by side.
    """
    # Ensure the inputs are lists of the correct length
    if titles is None:
        titles = ["Matrix 1", "Matrix 2", "Matrix 3"]
    if colormaps is None:
        colormaps = [None, None, None]
    if x_labels is None:
        x_labels = ["Columns", "Columns", "Columns"]
    if y_labels is None:
        y_labels = ["Rows", "Rows", "Rows"]

    # Set up the figure and subplots
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Set the overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=24, fontweight="bold")

    if same_scale:
        # Calculate the global minimum and maximum for color scaling
        all_data = np.concatenate(
            (matrix1.flatten(), matrix2.flatten(), matrix3.flatten())
        )
        vmin = all_data.min()
        vmax = all_data.max()

        # Ensure central color is at value 0

    norm = [[], [], []]
    norm_mins = [[], [], []]
    norm_maxs = [[], [], []]
    matrices = [matrix1, matrix2, matrix3]
    for i in range(3):
        if same_scale:
            norm_min = np.min((vmin, 0))
            norm_max = np.max((vmax, 0))
            norm_center = np.max((vmin, 0))
            norm[i] = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)
        else:
            flattened_matrix = np.array(matrices[i]).flatten()
            matrix_min = flattened_matrix.min()
            matrix_max = flattened_matrix.max()
            norm_min = np.min((matrix_min, 0.0))
            norm_max = np.max((matrix_max, 0.0))
            norm_center = np.max((matrix_min, 0.0))

        if norm_max == norm_center and norm_min == norm_center:
            norm_min -= 1e-10
            norm_max += 1e-10

        if norm_min == norm_center:
            norm_min -= np.abs(norm_max - norm_center) / 10000

        if norm_max == norm_center:
            norm_max += np.abs(norm_min - norm_center) / 10000

        norm_mins[i] = norm_min
        norm_maxs[i] = norm_max
        norm[i] = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)

    # Plot each matrix
    im1 = axes[0].imshow(matrix1, cmap=colormaps[0], norm=norm[0], interpolation="none", aspect=aspect)
    axes[0].set_title(titles[0], fontsize=24)
    axes[0].set_xlabel(x_labels[0], fontsize=24)
    axes[0].set_ylabel(y_labels[0], fontsize=24)
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=number_size)
    if exact_color_ticks:
        cbar.set_ticks([norm_mins[0], norm_mins[0]/2, 0, norm_maxs[0]/2, norm_maxs[0]])

    im2 = axes[1].imshow(matrix2, cmap=colormaps[1], norm=norm[1], interpolation="none", aspect=aspect)
    axes[1].set_title(titles[1], fontsize=24)
    axes[1].set_xlabel(x_labels[1], fontsize=24)
    axes[1].set_ylabel(y_labels[1], fontsize=24)
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=number_size)
    if exact_color_ticks:
        cbar.set_ticks([norm_mins[1], norm_mins[1]/2, 0, norm_maxs[1]/2, norm_maxs[1]])

    im3 = axes[2].imshow(matrix3, cmap=colormaps[2], norm=norm[2], interpolation="none", aspect=aspect)
    axes[2].set_title(titles[2], fontsize=24)
    axes[2].set_xlabel(x_labels[2], fontsize=24)
    axes[2].set_ylabel(y_labels[2], fontsize=24)
    axes[2].invert_yaxis()
    axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=number_size)
    if exact_color_ticks:
        cbar.set_ticks([norm_mins[2], norm_mins[2]/2, 0, norm_maxs[2]/2, norm_maxs[2]])
    
    if x_ticks is not None:
        axes[0].set_xticks(x_ticks)
        axes[1].set_xticks(x_ticks)
        axes[2].set_xticks(x_ticks)
    if y_ticks is not None:
        axes[0].set_yticks(y_ticks)
        axes[1].set_yticks(y_ticks)
        axes[2].set_yticks(y_ticks)
        
    axes[0].tick_params(axis='both', which='major', labelsize=number_size)
    axes[1].tick_params(axis='both', which='major', labelsize=number_size)
    axes[2].tick_params(axis='both', which='major', labelsize=number_size)

    # Adjust spacing between plots
    plt.tight_layout()


def plot_two_matrices_side_by_side(
    matrix1,
    matrix2,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False,
    fig=None,
    axes=None,
    x_ticks=None,
    y_ticks=None,
):
    """
    Plots two 2D matrices side by side.

    Note color scheme is set so that there is the same number of colors for positive and negative values (so for some consistency, albeit scale of colorbar is going to be variable as a result)

    Parameters:
    - matrix1, matrix2: 2D numpy arrays to be plotted.
    - overall_title: The title for the entire figure. Default is None.
    - titles: A list of titles for the subplots. Default is None.
    - colormaps: A list of colormaps for the subplots. Default is None.
    - x_labels: A list of x-axis labels for the subplots. Default is None.
    - y_labels: A list of y-axis labels for the subplots. Default is None.
    - same_scale: A boolean indicating whether to use the same color scale for all plots
    - x_ticks: Shared X ticks to set for the plots
    - y_ticks: Shared Y ticks to set for the plots
    """
    # Ensure the inputs are lists of the correct length
    if titles is None:
        titles = ["Matrix 1", "Matrix 2"]
    if colormaps is None:
        colormaps = [None, None, None]
    if x_labels is None:
        x_labels = ["Columns", "Columns"]
    if y_labels is None:
        y_labels = ["Rows", "Rows"]

    # Set up the figure and subplots
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Set the overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=24, fontweight="bold")

    if same_scale:
        # Calculate the global minimum and maximum for color scaling
        all_data = np.concatenate((matrix1.flatten(), matrix2.flatten()))
        vmin = all_data.min()
        vmax = all_data.max()

        # Ensure central color is at value 0

    norm = [[], [], []]
    matrices = [matrix1, matrix2]
    for i in range(2):
        if same_scale:
            norm_min = np.min((vmin, 0))
            norm_max = np.max((vmax, 0))
            norm_center = np.max((vmin, 0))
            norm[i] = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)
        else:
            flattened_matrix = np.array(matrices[i]).flatten()
            matrix_min = flattened_matrix.min()
            matrix_max = flattened_matrix.max()
            norm_min = np.min((matrix_min, 0.0))
            norm_max = np.max((matrix_max, 0.0))
            norm_center = np.max((matrix_min, 0.0))

        if norm_max == norm_center and norm_min == norm_center:
            norm_min -= 1e-10
            norm_max += 1e-10

        if norm_min == norm_center:
            norm_min -= np.abs(norm_max - norm_center) / 10000

        if norm_max == norm_center:
            norm_max += np.abs(norm_min - norm_center) / 10000

        norm[i] = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)

    # Plot each matrix
    im1 = axes[0].imshow(matrix1, cmap=colormaps[0], norm=norm[0], interpolation="none")
    axes[0].set_title(titles[0], fontsize=24)
    axes[0].set_xlabel(x_labels[0], fontsize=24)
    axes[0].set_ylabel(y_labels[0], fontsize=24)
    axes[0].invert_yaxis()
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(matrix2, cmap=colormaps[1], norm=norm[1], interpolation="none")
    axes[1].set_title(titles[1], fontsize=24)
    axes[1].set_xlabel(x_labels[1], fontsize=24)
    axes[1].set_ylabel(y_labels[1], fontsize=24)
    axes[1].invert_yaxis()
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    if x_ticks is not None:
        axes[0].set_xticks(x_ticks)
        axes[1].set_xticks(x_ticks)
    if y_ticks is not None:
        axes[0].set_yticks(y_ticks)
        axes[1].set_yticks(y_ticks)

    # Adjust spacing between plots
    plt.tight_layout()


def collapse_resolution_indices(data, i):
    # Assumes data indexed as data[l][i][j][k] with same dimensions for each [l]
    # Intended for Y or L (with i,j,k indicating wavelet coefficient indices)
    # Extract matrices for a fixed resolution i (assumes for each )
    indices = []
    li_data = data[0][i]
    for j, lij_data in enumerate(li_data):
        for k, lijk_data in enumerate(lij_data):
            indices.append((j, k))
    n_data_points = len(indices)
    n_features = len(data)

    matrix = np.zeros((n_data_points, n_features))
    for l, l_data in enumerate(data):
        for idx, (j, k) in enumerate(indices):
            matrix[idx, l] = l_data[i][j][k]
    return matrix


def generate_resolution_tuples(data1, data2, data3):
    # Assumes data indexed as data[l][i][j][k] with same dimensions for each [l]
    # Intended for Y or L (with i,j,k indicating wavelet coefficient indices)
    result = []
    n_resolutions = len(data1[0])

    for i in range(n_resolutions):
        matrix1 = collapse_resolution_indices(data1, i)
        matrix2 = collapse_resolution_indices(data2, i)
        matrix3 = collapse_resolution_indices(data3, i)
        result.append((matrix1, matrix2, matrix3))

    return result


def plot_wavelet_matrices_side_by_side(
    data1,
    data2,
    data3,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False,
    aspect=None
):
    # The same_scale apply only on a per row basis
    resolution_tuples = generate_resolution_tuples(data1, data2, data3)
    n_resolutions = len(resolution_tuples)
    fig, axes = plt.subplots(n_resolutions, 3, figsize=(15, 5 * n_resolutions))
    for idx, (res_data1, res_data2, res_data3) in enumerate(resolution_tuples):
        plot_matrices_side_by_side(
            res_data1,
            res_data2,
            res_data3,
            overall_title=overall_title if idx == 0 else None,
            titles=titles if idx == 0 else ["", "", ""],
            colormaps=colormaps,
            x_labels=x_labels,
            y_labels=y_labels,
            same_scale=same_scale,
            fig=fig,
            axes=axes[idx],
            aspect=aspect
        )


def plot_two_spot_loadings_side_by_side(
    data1,
    data2,
    feature_index,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False,
    x_ticks=None,
    y_ticks=None,
):
    # Intended for plotting Y for specific feature or L for specific factor all in spot space, side by side
    matrix1 = data1[feature_index]
    matrix2 = data2[feature_index]
    plot_two_matrices_side_by_side(
        matrix1,
        matrix2,
        overall_title,
        titles,
        colormaps,
        x_labels,
        y_labels,
        same_scale,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
    )


def plot_spot_loadings_side_by_side(
    data1,
    data2,
    data3,
    feature_index,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False
):
    # Intended for plotting Y for specific feature or L for specific factor all in spot space, side by side
    matrix1 = data1[feature_index]
    matrix2 = data2[feature_index]
    matrix3 = data3[feature_index]
    plot_matrices_side_by_side(
        matrix1,
        matrix2,
        matrix3,
        overall_title,
        titles,
        colormaps,
        x_labels,
        y_labels,
        same_scale,
    )


def plot_matrix(
    matrix,
    overall_title=None,
    title=None,
    colormap=None,
    x_label=None,
    y_label=None,
    fig=None,
    ax=None,
    x_ticks=None,
    y_ticks=None,
    exact_color_ticks=False
):
    """
    Plots a 2D matrices

    Note color scheme is set so that there is the same number of colors for positive and negative values (so for some consistency, albeit scale of colorbar is going to be variable as a result)

    Parameters:
    - matrix: 2D numpy array to be plotted.
    - overall_title: The title for the entire figure. Default is None.
    - title: A title for the plot. Default is None.
    - colormap: A colormap for the plot. Default is None.
    - x_label: A x-axis label for the plot. Default is None.
    - y_label: A y-axis label for the plot. Default is None.
    """
    # Ensure the inputs are lists of the correct length
    if title is None:
        title = None
    if colormap is None:
        colormap = None
    if x_label is None:
        x_label = "Columns"
    if y_label is None:
        y_label = "Rows"

    # Set up the figure and subplots
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # Set the overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=24, fontweight="bold")
        
    if title is not None:
        ax.set_title(title, fontsize=24)

    flattened_matrix = np.array(matrix).flatten()
    matrix_min = flattened_matrix.min()
    matrix_max = flattened_matrix.max()
    norm_min = np.min((matrix_min, 0.0))
    norm_max = np.max((matrix_max, 0.0))
    norm_center = np.max((matrix_min, 0.0))

    if norm_max == norm_center and norm_min == norm_center:
        norm_min -= 1e-10
        norm_max += 1e-10

    if norm_min == norm_center:
        norm_min -= np.abs(norm_max - norm_center) / 10000

    if norm_max == norm_center:
        norm_max += np.abs(norm_min - norm_center) / 10000

    norm = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)

    # Plot each matrix
    im1 = ax.imshow(matrix, cmap=colormap, norm=norm, interpolation="none")
    ax.set_title(title, fontsize=24)
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=16)
    if exact_color_ticks:
        cbar.set_ticks([norm_min, norm_min/2, 0, norm_max/2, norm_max])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust spacing between plots
    plt.tight_layout()


def plot_spot_loadings(
    data,
    feature_index,
    overall_title=None,
    titles=None,
    colormaps=None,
    x_labels=None,
    y_labels=None,
    same_scale=False,
):
    # Intended for plotting Y for specific feature or L for specific factor all in spot space
    matrix = data[feature_index]
    plot_matrix(
        matrix,
        overall_title,
        titles,
        colormaps,
        x_labels,
        y_labels,
        same_scale,
    )


## Plotting L in wavelet space


def flatten_to_square(vec):
    """Expects vec to be a list"""
    n = len(vec)
    dim = int(math.sqrt(n))

    if dim * dim != n:
        raise ValueError("The length of the input list must be a perfect square.")

    return np.array([vec[i * dim : (i + 1) * dim] for i in range(dim)])


def expand_matrix(matrix, factor=2):
    """Function to expand a matrix so that individual every element is expanded onto a factor * factor grid"""
    return np.kron(matrix, np.ones((factor, factor)))


def plot_wavelet_L(
    L_elements,
    n_spots,
    resolution,
    n_resolutions,
    fig,
    ax,
    colormap=None,
    x_label="Horizontal ordinate",
    y_label="Vertical ordinate",
    x_ticks=None,
    y_ticks=None,
    exact_color_ticks=False
):
    "Plots the inferred L in wavelet space overlaid onto the corresponding spot space region that the L was computed over, expects an innermost substructure in L, i.e. if L indexed by l,i,j,k then expects a substructure of L with fixed l,i,j"
    "n_spots is the number of spots (should be a power of 4) in the original spot space data"
    "resolution is the resolution index of the L_elements, i.e. the i if L is indexed by l,i,j,k starting with i=0 as approx level, i=1 as coarsest detail level, i=2 as the second most coarsest detail level, and so on"

    # Reformat and expand L_elements in preparation of plotting onto the spot space
    if resolution == 0:
        wavelet_n_spots = int(n_spots / (4 ** (n_resolutions - 1)))
    else:
        wavelet_n_spots = int(n_spots / (4 ** (n_resolutions - resolution)))

    assert wavelet_n_spots == len(L_elements)

    wavelet_spots_per_region = int(n_spots / wavelet_n_spots)
    wavelet_region_side_length = int(math.sqrt(wavelet_spots_per_region))
    L_matrix = flatten_to_square(L_elements)
    L_matrix_expanded = expand_matrix(L_matrix, factor=wavelet_region_side_length)

    # Helper variables to help set color scale
    matrix_min = L_matrix_expanded.min()
    matrix_max = L_matrix_expanded.max()
    norm_min = np.min((matrix_min, 0.0))
    norm_max = np.max((matrix_max, 0.0))
    norm_center = np.max((matrix_min, 0.0))
    if norm_max == norm_center and norm_min == norm_center:
        norm_min -= 1e-10
        norm_max += 1e-10

    if norm_min == norm_center:
        norm_min -= np.abs(norm_max - norm_center) / 10000

    if norm_max == norm_center:
        norm_max += np.abs(norm_min - norm_center) / 10000
    norm = TwoSlopeNorm(vmin=norm_min, vcenter=norm_center, vmax=norm_max)

    # Plot matrix
    im = ax.imshow(
        L_matrix_expanded, cmap=colormap, aspect='equal', norm=norm, interpolation="none"
    )
    ax.set_xlabel(x_label, fontsize=36)
    ax.set_ylabel(y_label, fontsize=36)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=24)
    if exact_color_ticks:
        cbar.set_ticks([norm_min, norm_min/2, 0, norm_max/2, norm_max])
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)
    ax.tick_params(axis='both', which='major', labelsize=24)


def plot_all_wavelet_L(
    L_factor,
    n_resolutions,
    n_spots,
    figsize_per_plot=(5, 5),
    colormap=None,
    overall_title=None,
):
    "Plots the inferred L_factor (first level substructure in L objects) in wavelet space for each individual resolution"
    figsize = (figsize_per_plot[0] * 3, figsize_per_plot[1] * n_resolutions)
    fig, axes = plt.subplots(n_resolutions, 3, figsize=figsize)

    # Set the overall title if provided
    if overall_title is not None:
        fig.suptitle(overall_title, fontsize=24, fontweight="bold")

    # Plot approx level coefficients
    ax = axes[0, 0]
    plot_wavelet_L(L_factor[0][0], n_spots, 0, n_resolutions, fig, ax, colormap)
    ax.set_title("approx")
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")

    # Plot detail level coefficients
    for i in range(1, n_resolutions):
        for j in range(3):
            ax = axes[i, j]
            plot_wavelet_L(L_factor[i][j], n_spots, i, n_resolutions, fig, ax, colormap)
            ax.set_title(f"detail({i}, {j})")
