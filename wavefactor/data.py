"""
Data ingestion, spatial rasterization, and 2D wavelet decomposition for WaveFactor.
"""

from typing import Union, Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import pywt


def is_power_of_4(n: int) -> bool:
    """Return True if n is an exact power of 4."""
    try:
        n = int(n)
    except (ValueError, TypeError):
        return False
    return n > 0 and (n & (n - 1)) == 0 and (n.bit_length() - 1) % 2 == 0


def get_L_shape(n_spots: int, n_resolutions: int, n_factors: int) -> List:
    """Generates empty nested list structure matching L shape."""
    return [
        [[np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - 1)))))]]
        + [
            [
                np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - i - 1)))))
                for _ in range(3)
            ]
            for i in range(n_resolutions - 1)
        ]
        for _ in range(n_factors)
    ]


def get_Y_shape(n_spots: int, n_resolutions: int, n_features: int) -> List:
    """Generates empty nested list structure matching Y shape."""
    return [
        [[np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - 1)))))]]
        + [
            [
                np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - i - 1)))))
                for _ in range(3)
            ]
            for i in range(n_resolutions - 1)
        ]
        for _ in range(n_features)
    ]


class WaveFactorData:
    """
    Container holding processed spatial transcriptomics matrices,
    2D discrete wavelet decompositions (true_Y), dimensions, and spatial mappings.

    Attributes
    ----------
    true_Y : List
        Nested DWT wavelet decomposition matrices for all features.
    dimensions : Dict[str, Any]
        Dictionary with keys 'n_spots', 'n_resolutions', 'n_factors', 'n_features'.
    grid_side_length : int
        Side length L of the square L x L spatial grid (e.g. 64 for a 64x64 grid).
    grid_shape : Tuple[int, int]
        2D grid shape (grid_side_length, grid_side_length).
    spot_grid_coords : np.ndarray
        Array of shape (N_spots, 2) holding Cartesian [grid_x, grid_y] coordinates:
        - Column 0: grid_x (horizontal coordinate, column index in 0..L-1)
        - Column 1: grid_y (vertical coordinate, row index in 0..L-1)
    grid_x : np.ndarray
        1D integer grid x-coordinates (columns in 0..L-1).
    grid_y : np.ndarray
        1D integer grid y-coordinates (rows in 0..L-1).
    coords : np.ndarray, optional
        Original raw spatial coordinates (N_spots, 2).
    """

    def __init__(
        self,
        true_Y: List,
        dimensions: Dict[str, Any],
        grid_side_length: int,
        spot_grid_coords: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
    ):
        self.true_Y = true_Y
        self.dimensions = dimensions
        self.grid_side_length = int(grid_side_length)
        self.spot_grid_coords = spot_grid_coords
        self.coords = coords

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """2D spatial grid shape (grid_side_length, grid_side_length)."""
        return (self.grid_side_length, self.grid_side_length)

    @property
    def grid_x(self) -> Optional[np.ndarray]:
        """1D integer grid x-coordinates (columns in 0..L-1)."""
        return self.spot_grid_coords[:, 0] if self.spot_grid_coords is not None else None

    @property
    def grid_y(self) -> Optional[np.ndarray]:
        """1D integer grid y-coordinates (rows in 0..L-1)."""
        return self.spot_grid_coords[:, 1] if self.spot_grid_coords is not None else None


def prepare_spatial_data(
    X: np.ndarray,
    coords: np.ndarray,
    n_factors: int = 10,
    n_length_scales: int = 4,
) -> WaveFactorData:
    """
    Preprocesses spatial transcriptomics input data, validates 2D spatial grid mappings,
    and executes 2D DWT wavelet decomposition.

    Parameters
    ----------
    X : np.ndarray
        Gene expression matrix of shape (N_spots, N_features).
    coords : np.ndarray
        Spatial (x, y) integer grid coordinates for each spot, shape (N_spots, 2).
        Must be a unique 1-to-1 mapping onto the [0, L-1] x [0, L-1] lattice,
        where N_spots = L * L is an exact power of 4.
    n_factors : int, default=10
        Number of latent factors (K).
    n_length_scales : int, default=4
        Number of wavelet detail levels (D). Total resolutions R = D + 1.
        Must satisfy 2^D <= L (i.e. D <= log2(L)).

    Returns
    -------
    WaveFactorData
        Structured data container with wavelet decomposition and dimension constants.
    """
    # 1. Parse and validate 2D expression matrix
    expr_matrix = np.asarray(X, dtype=np.float64)
    if expr_matrix.ndim != 2:
        raise ValueError(
            f"X must be a 2D numpy array of shape (N_spots, N_features), got ndim={expr_matrix.ndim} with shape {expr_matrix.shape}."
        )

    N_spots, N_features = expr_matrix.shape

    # 2. Strict power-of-4 spot count verification
    if not is_power_of_4(N_spots):
        raise ValueError(
            f"Number of spots (N={N_spots}) must be an exact power of 4 (e.g. 64, 256, 1024, 4096) "
            "to form a complete 2D dyadic wavelet lattice."
        )

    side_len = int(round(np.sqrt(N_spots)))
    max_D = int(round(np.log2(side_len)))

    # 3. Wavelet resolution bound check (2^D <= L)
    if n_length_scales > max_D:
        raise ValueError(
            f"n_length_scales (D={n_length_scales}) cannot exceed log2(grid_side_length)={max_D} "
            f"for a grid of side length L={side_len} (N={N_spots} spots)."
        )
    if n_length_scales <= 0:
        raise ValueError(f"n_length_scales must be a positive integer, got {n_length_scales}.")

    # 4. Parse and strictly validate spatial coordinates
    spot_coords = np.asarray(coords)
    if spot_coords.ndim != 2 or spot_coords.shape != (N_spots, 2):
        raise ValueError(
            f"coords must be a 2D array of shape (N_spots, 2) = ({N_spots}, 2), got shape {spot_coords.shape}."
        )

    # Validate integer coordinates
    if not (np.issubdtype(spot_coords.dtype, np.integer) or np.all(np.equal(np.mod(spot_coords, 1), 0))):
        raise ValueError(
            f"coords must contain integer grid indices on [0, {side_len - 1}] x [0, {side_len - 1}]."
        )

    grid_x = spot_coords[:, 0].astype(int)
    grid_y = spot_coords[:, 1].astype(int)

    # Bounds check
    if grid_x.min() < 0 or grid_x.max() >= side_len or grid_y.min() < 0 or grid_y.max() >= side_len:
        raise ValueError(
            f"coords indices out of bounds: all x and y must be within [0, {side_len - 1}] "
            f"for an inferred {side_len}x{side_len} grid."
        )

    # Bijection check: every lattice cell must be covered exactly once
    flat_indices = grid_y * side_len + grid_x
    if len(np.unique(flat_indices)) != N_spots:
        raise ValueError(
            f"Input coords must define a unique 1-to-1 lattice mapping onto the {side_len}x{side_len} grid. "
            "Found duplicate or missing grid cell coordinates."
        )

    spot_grid_coords = np.column_stack([grid_x, grid_y])

    # 5. Direct 1-to-1 lattice feature grid assignment
    feature_grid = np.zeros((side_len, side_len, N_features), dtype=np.float64)
    feature_grid[grid_y, grid_x, :] = expr_matrix

    # Compute 2D DWT using Haar basis
    true_Y = _compute_2d_dwt(feature_grid, n_length_scales, wavelet="haar")
    total_grid_spots = side_len * side_len
    dimensions = _build_dimensions_dict(
        n_spots=total_grid_spots,
        n_length_scales=n_length_scales,
        n_factors=n_factors,
        n_features=N_features,
    )

    return WaveFactorData(
        true_Y=true_Y,
        dimensions=dimensions,
        grid_side_length=side_len,
        spot_grid_coords=spot_grid_coords,
        coords=spot_coords,
    )


def _compute_2d_dwt(feature_grid: np.ndarray, n_length_scales: int, wavelet: str = "haar") -> List:
    """Performs 2D DWT across all features and formats into true_Y nested structure."""
    H, W, G = feature_grid.shape
    flattened_wavelet_matrices = []

    for g in range(G):
        coeffs = pywt.wavedec2(feature_grid[:, :, g], wavelet, level=n_length_scales)
        flattened_coeffs = []
        flattened_coeffs.append((coeffs[0].flatten(),))
        for res in coeffs[1:]:
            flattened_coeffs.append(tuple(matrix.flatten() for matrix in res))
        flattened_wavelet_matrices.append(flattened_coeffs)

    return flattened_wavelet_matrices


def _build_dimensions_dict(n_spots: int, n_length_scales: int, n_factors: int, n_features: int) -> Dict[str, Any]:
    """Constructs dimensions dictionary matching C++ CAVI interface."""
    n_resolutions = n_length_scales + 1
    return {
        "n_factors": n_factors,
        "n_resolutions": n_resolutions,
        "n_features": n_features,
        "n_spots": n_spots,
        "L_shape": get_L_shape(n_spots, n_resolutions, n_factors),
        "Y_shape": get_Y_shape(n_spots, n_resolutions, n_features),
        "p_pi_shape": (n_resolutions,),
        "ab_t_shape": (n_resolutions, n_factors),
        "ab_tau_shape": (n_resolutions, n_features),
        "F_shape": (n_factors, n_features),
    }
