"""
Unit tests for wavefactor.data (spatial data preparation, gridding, and 2D DWT).
"""

from tests.compat import pytest
import numpy as np
import pandas as pd
from wavefactor.data import prepare_spatial_data, is_power_of_4, get_L_shape, get_Y_shape


def test_is_power_of_4():
    assert is_power_of_4(1)
    assert is_power_of_4(4)
    assert is_power_of_4(16)
    assert is_power_of_4(64)
    assert is_power_of_4(256)
    assert is_power_of_4(1024)
    assert is_power_of_4(4096)
    assert not is_power_of_4(0)
    assert not is_power_of_4(2)
    assert not is_power_of_4(8)
    assert not is_power_of_4(32)
    assert not is_power_of_4(100)


def test_prepare_spatial_data_valid_lattice():
    np.random.seed(42)
    L = 8
    N_spots = L * L  # 64 (exact power of 4)
    N_genes = 5
    expr = np.random.randn(N_spots, N_genes)

    # Construct canonical grid coordinates (64, 2)
    xx, yy = np.meshgrid(np.arange(L), np.arange(L))
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    data = prepare_spatial_data(
        X=expr,
        coords=coords,
        n_factors=3,
        n_length_scales=2,
    )

    assert data.grid_side_length == 8
    assert data.grid_shape == (8, 8)
    assert data.dimensions["n_spots"] == 64
    assert data.dimensions["n_resolutions"] == 3
    assert data.dimensions["n_factors"] == 3
    assert data.dimensions["n_features"] == N_genes
    assert len(data.true_Y) == N_genes
    assert len(data.true_Y[0]) == 3  # R=3 resolutions
    assert data.spot_grid_coords.shape == (N_spots, 2)
    assert np.array_equal(data.grid_x, coords[:, 0])
    assert np.array_equal(data.grid_y, coords[:, 1])


def test_prepare_spatial_data_non_power_of_4_raises():
    np.random.seed(42)
    N_spots = 100  # Not a power of 4
    expr = np.random.randn(N_spots, 4)
    coords = np.zeros((N_spots, 2), dtype=int)

    with pytest.raises(ValueError, match="exact power of 4"):
        prepare_spatial_data(X=expr, coords=coords, n_factors=2, n_length_scales=2)


def test_prepare_spatial_data_resolution_overflow_raises():
    np.random.seed(42)
    L = 8
    N_spots = 64  # max D = log2(8) = 3
    expr = np.random.randn(N_spots, 4)
    xx, yy = np.meshgrid(np.arange(L), np.arange(L))
    coords = np.column_stack([xx.ravel(), yy.ravel()])

    with pytest.raises(ValueError, match="cannot exceed log2"):
        prepare_spatial_data(X=expr, coords=coords, n_factors=2, n_length_scales=4)


def test_prepare_spatial_data_duplicate_coords_raises():
    np.random.seed(42)
    L = 8
    N_spots = 64
    expr = np.random.randn(N_spots, 4)
    xx, yy = np.meshgrid(np.arange(L), np.arange(L))
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    # Introduce duplicate coordinate (overwrite last with first)
    coords[-1] = coords[0]

    with pytest.raises(ValueError, match="unique 1-to-1 lattice"):
        prepare_spatial_data(X=expr, coords=coords, n_factors=2, n_length_scales=2)


def test_prepare_spatial_data_out_of_bounds_coords_raises():
    np.random.seed(42)
    L = 8
    N_spots = 64
    expr = np.random.randn(N_spots, 4)
    xx, yy = np.meshgrid(np.arange(L), np.arange(L))
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    coords[0] = [8, 0]  # 8 is out of bounds for L=8 (valid: 0..7)

    with pytest.raises(ValueError, match="out of bounds"):
        prepare_spatial_data(X=expr, coords=coords, n_factors=2, n_length_scales=2)


def test_prepare_spatial_data_from_3d_grid_raises():
    np.random.seed(42)
    H, W, G = 16, 16, 4
    grid_data = np.random.randn(H, W, G)
    coords = np.zeros((H * W, 2), dtype=int)
    with pytest.raises(ValueError):
        prepare_spatial_data(
            X=grid_data,
            coords=coords,
            n_factors=2,
            n_length_scales=2,
        )
