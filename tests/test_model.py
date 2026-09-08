"""
Integration tests for WaveFactor estimator and WaveFactorResult.
"""

from tests.compat import pytest
import numpy as np
from wavefactor import WaveFactor, WaveFactorResult


def test_wavefactor_estimator_fit_transform():
    np.random.seed(123)
    # Generate synthetic spatial data on 8x8 = 64 spots
    N_spots = 64
    N_genes = 6
    K = 2
    D = 2  # R = 3

    # Generate 2 spatial patterns on 8x8 integer grid
    gx = np.arange(8)
    gy = np.arange(8)
    xx, yy = np.meshgrid(gx, gy)
    pattern1 = np.sin(2 * np.pi * (xx / 8.0)).ravel()
    pattern2 = np.cos(2 * np.pi * (yy / 8.0)).ravel()

    true_S = np.column_stack([pattern1, pattern2])  # (64, 2)
    true_F = np.random.randn(K, N_genes)  # (2, 6)

    noise = np.random.randn(N_spots, N_genes) * 0.1
    expr = np.dot(true_S, true_F) + noise

    coords = np.column_stack([xx.ravel(), yy.ravel()])

    # Fit WaveFactor
    model = WaveFactor(
        n_factors=K,
        n_length_scales=D,
        spatial_prior=np.array([0.9, 0.5, 0.2]),
        gene_prior=np.full((K, N_genes), 0.5),
        max_iter=15,
        tol=1e-5,
        n_init=2,
        n_jobs=1,
        random_state=42,
        verbose=False,
    )

    factors = model.fit_transform(expr, coords)

    assert factors.shape == (N_spots, K)
    assert np.array_equal(factors, model.transform())
    assert model.factors_.shape == (N_spots, K)
    assert model.loadings_.shape == (K, N_genes)
    assert len(model.elbo_history_) > 0
    assert np.isfinite(model.elbo_)

    # Test result object
    res = model.get_result()
    assert isinstance(res, WaveFactorResult)
    assert res.n_factors == K
    assert res.n_features == N_genes
    assert res.grid_side_length == 8
    assert res.grid_shape == (8, 8)
    assert res.spatial_factor_maps.shape == (8, 8, K)
    assert res.spot_grid_coords.shape == (N_spots, 2)
    assert np.array_equal(res.grid_x, res.spot_grid_coords[:, 0])
    assert np.array_equal(res.grid_y, res.spot_grid_coords[:, 1])
    assert res.gene_pip.shape == (K, N_genes)
    # Test full variational distribution properties
    assert isinstance(res.mu_L, list)
    assert isinstance(res.sigma_squared_L, list)
    assert res.mu_F.shape == (K, N_genes)
    assert res.sigma_squared_F.shape == (K, N_genes)
    assert res.loadings_variance.shape == (K, N_genes)
    assert res.alpha_hat_t.shape == (D + 1, K)
    assert res.beta_hat_t.shape == (D + 1, K)
    assert res.alpha_hat_tau.shape == (D + 1, N_genes)
    assert res.beta_hat_tau.shape == (D + 1, N_genes)
    assert np.all(res.loadings_variance >= 0.0)

    # Test RuntimeError on unfitted model
    unfitted = WaveFactor()
    with pytest.raises(RuntimeError):
        unfitted.transform()


def test_estimator_get_set_params():
    model = WaveFactor(n_factors=5, n_length_scales=3, max_iter=200)
    params = model.get_params()
    assert params["n_factors"] == 5
    assert params["n_length_scales"] == 3
    assert params["max_iter"] == 200

    model.set_params(n_factors=8, max_iter=500)
    assert model.n_factors == 8
    assert model.max_iter == 500
