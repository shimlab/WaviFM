"""
Unit tests for wavefactor.priors.
"""

from tests.compat import pytest
import numpy as np
from wavefactor.priors import Priors
from wavefactor.data import _build_dimensions_dict


def get_mock_dimensions(n_spots=64, n_length_scales=2, n_factors=3, n_features=4):
    return _build_dimensions_dict(
        n_spots=n_spots,
        n_length_scales=n_length_scales,
        n_factors=n_factors,
        n_features=n_features,
    )


def test_default_canonical_priors():
    dims = get_mock_dimensions()
    priors = Priors()
    priors_dict = priors.build_priors_dict(dims)

    assert "log_p_pi" in priors_dict
    assert "log_p_eta" in priors_dict
    assert "alpha_t" in priors_dict
    assert "beta_t" in priors_dict
    assert "alpha_tau" in priors_dict
    assert "beta_tau" in priors_dict

    # Check shapes
    assert priors_dict["log_p_pi"].shape == dims["p_pi_shape"]
    assert priors_dict["log_p_eta"].shape == dims["F_shape"]
    assert priors_dict["alpha_t"].shape == dims["ab_t_shape"]
    assert priors_dict["beta_t"].shape == dims["ab_t_shape"]
    assert priors_dict["alpha_tau"].shape == dims["ab_tau_shape"]
    assert priors_dict["beta_tau"].shape == dims["ab_tau_shape"]

    # Baseline defaults
    assert np.allclose(np.exp(priors_dict["log_p_pi"]), 0.5)
    assert np.allclose(np.exp(priors_dict["log_p_eta"]), 0.5)
    assert np.allclose(priors_dict["alpha_t"], 1.0)
    assert np.allclose(priors_dict["beta_t"], 1.0)
    assert np.allclose(priors_dict["alpha_tau"], 1.0)
    assert np.allclose(priors_dict["beta_tau"], 1.0)


def test_custom_array_priors():
    dims = get_mock_dimensions(n_length_scales=2, n_factors=3, n_features=4)
    R = dims["n_resolutions"]  # 3
    K = dims["n_factors"]      # 3
    G = dims["n_features"]     # 4

    custom_pi = np.array([0.9, 0.6, 0.2])
    custom_eta = np.full((K, G), 0.15)
    custom_alpha_t = np.full((R, K), 2.0)
    custom_beta_t = np.full((R, K), 1.5)
    custom_alpha_tau = np.full((R, G), 0.8)
    custom_beta_tau = np.full((R, G), 0.5)

    priors = Priors(
        spatial_prior=custom_pi,
        gene_prior=custom_eta,
        alpha_t=custom_alpha_t,
        beta_t=custom_beta_t,
        alpha_tau=custom_alpha_tau,
        beta_tau=custom_beta_tau,
    )
    p_dict = priors.build_priors_dict(dims)

    assert np.allclose(np.exp(p_dict["log_p_pi"]), custom_pi)
    assert np.allclose(np.exp(p_dict["log_p_eta"]), custom_eta)
    assert np.allclose(p_dict["alpha_t"], custom_alpha_t)
    assert np.allclose(p_dict["beta_t"], custom_beta_t)
    assert np.allclose(p_dict["alpha_tau"], custom_alpha_tau)
    assert np.allclose(p_dict["beta_tau"], custom_beta_tau)


def test_invalid_priors_raise():
    dims = get_mock_dimensions(n_length_scales=2, n_factors=3, n_features=4)

    # Spatial prior must be 1D array of correct length
    with pytest.raises(Exception):
        Priors(spatial_prior=np.array([[0.5, 0.5], [0.5, 0.5]])).build_priors_dict(dims)

    # Spatial prior values must be in (0, 1)
    with pytest.raises(ValueError):
        Priors(spatial_prior=np.array([1.5, 0.5, 0.5])).build_priors_dict(dims)

    with pytest.raises(ValueError):
        Priors(spatial_prior=np.array([-0.1, 0.5, 0.5])).build_priors_dict(dims)

    # Gene prior must be 2D array of correct shape
    with pytest.raises(Exception):
        Priors(gene_prior=np.array([0.5, 0.5])).build_priors_dict(dims)

    # Gene prior values must be in (0, 1)
    with pytest.raises(ValueError):
        Priors(gene_prior=np.full((3, 4), 1.0)).build_priors_dict(dims)

    # Precision priors must be strictly positive
    with pytest.raises(ValueError):
        Priors(alpha_t=np.full((3, 3), -1.0)).build_priors_dict(dims)
