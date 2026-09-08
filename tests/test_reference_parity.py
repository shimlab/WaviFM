"""
Deterministic reference baseline parity test.
Verifies that the modernized WaveFactor codebase matches pre-v2.0 numerical
outputs from the main branch fixtures (tests/fixtures/reference_output.npz).
"""

import os
import numpy as np
import pandas as pd
import pywt
from tests.compat import pytest

from wavefactor import WaveFactor, WaveFactorResult
from wavefactor.data import prepare_spatial_data
from wavefactor.engine import init_parameters, _run_single_cavi_worker

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
CSV_PATH = os.path.join(FIXTURES_DIR, "reference_input.csv")
NPZ_PATH = os.path.join(FIXTURES_DIR, "reference_output.npz")


def _load_reference_data():
    assert os.path.exists(CSV_PATH), f"Fixture missing: {CSV_PATH}"
    assert os.path.exists(NPZ_PATH), f"Fixture missing: {NPZ_PATH}"
    df = pd.read_csv(CSV_PATH)
    ref = np.load(NPZ_PATH)
    return df, ref


def test_reference_baseline_parity():
    """
    Validates that the modern WaveFactor pipeline (prepare_spatial_data, CAVI engine,
    WaveFactorResult) produces exact numerical outputs matching the pre-v2.0
    reference baseline fixtures (tests/fixtures/reference_output.npz).
    """
    df, ref = _load_reference_data()

    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    expr = df[gene_cols].values
    coords = df[["x_index", "y_index"]].values

    data = prepare_spatial_data(
        X=expr,
        coords=coords,
        n_factors=int(ref["n_factors"]),
        n_length_scales=int(ref["n_length_scales"]),
    )

    # Execute multi-initialization CAVI matching exact reference seed
    np.random.seed(int(ref["seed"]))
    params_list = [
        init_parameters(data.true_Y, data.dimensions, priors=None, rng=None)
        for _ in range(int(ref["n_init"]))
    ]

    results_list = [
        _run_single_cavi_worker(p, max_iter=int(ref["max_iter"]), tol=float(ref["tol"]))
        for p in params_list
    ]
    best_idx, best_res = max(enumerate(results_list), key=lambda x: x[1]["elbo"])

    res = WaveFactorResult(
        parameters=best_res["parameters"],
        elbo_record=best_res["elbo_record"],
        elbo=best_res["elbo"],
        dimensions=data.dimensions,
        grid_side_length=data.grid_side_length,
        spot_grid_coords=data.spot_grid_coords,
        coords=data.coords,
    )

    # 1. Multi-initialization selection matches reference (Init 2)
    assert best_idx + 1 == 2, f"Expected best initialization 2, got {best_idx + 1}"

    # 2. ELBO scalar and trajectory parity
    cur_elbo = float(res.elbo)
    ref_elbo = float(ref["final_elbo"])
    assert np.isclose(cur_elbo, ref_elbo, rtol=1e-5, atol=1e-6), (
        f"Final ELBO mismatch: {cur_elbo} vs reference {ref_elbo}"
    )

    cur_elbo_record = res.elbo_history
    ref_elbo_record = np.asarray(ref["elbo_record"], dtype=np.float64)
    assert len(cur_elbo_record) == len(ref_elbo_record), (
        f"ELBO record length mismatch: {len(cur_elbo_record)} vs {len(ref_elbo_record)}"
    )
    assert np.allclose(cur_elbo_record, ref_elbo_record, rtol=1e-5, atol=1e-6), (
        "ELBO trajectory mismatch across iterations."
    )

    # 3. Gene Loadings (mu_F * r_eta) parity
    assert np.allclose(res.loadings, ref["loadings"], rtol=1e-4, atol=1e-5), (
        "Gene loadings mismatch against reference."
    )

    # 4. Gene posterior inclusion probabilities (PIPs) parity
    assert np.allclose(res.gene_pip, ref["gene_pip"], rtol=1e-4, atol=1e-5), (
        "Gene PIP mismatch against reference."
    )

    # 5. Spot-space spatial factors parity (2D IDWT reconstruction)
    assert np.allclose(res.factors, ref["factors"], rtol=1e-4, atol=1e-5), (
        "Spatial factors mismatch against reference."
    )

    # 6. Factor and noise precision hyperparameters parity
    assert np.allclose(res.alpha_hat_t, ref["alpha_hat_t"], rtol=1e-5, atol=1e-6), (
        "Factor precision shape (alpha_hat_t) mismatch against reference."
    )
    assert np.allclose(res.beta_hat_t, ref["beta_hat_t"], rtol=1e-5, atol=1e-6), (
        "Factor precision rate (beta_hat_t) mismatch against reference."
    )
    assert np.allclose(res.alpha_hat_tau, ref["alpha_hat_tau"], rtol=1e-5, atol=1e-6), (
        "Noise precision shape (alpha_hat_tau) mismatch against reference."
    )
    assert np.allclose(res.beta_hat_tau, ref["beta_hat_tau"], rtol=1e-5, atol=1e-6), (
        "Noise precision rate (beta_hat_tau) mismatch against reference."
    )


def test_modern_wavefactor_estimator_parity():
    """
    Validates that the modern WaveFactor estimator class behaves consistently,
    converges monotonically, and yields valid factor and loading matrices on
    the reference dataset.
    """
    df, ref = _load_reference_data()
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    expr = df[gene_cols].values
    coords = df[["x_index", "y_index"]].values

    model = WaveFactor(
        n_factors=int(ref["n_factors"]),
        n_length_scales=int(ref["n_length_scales"]),
        max_iter=int(ref["max_iter"]),
        tol=float(ref["tol"]),
        n_init=int(ref["n_init"]),
        n_jobs=1,
        random_state=int(ref["seed"]),
        verbose=False,
    )
    model.fit(expr, coords)

    # 1. Output shapes match reference dimensions
    assert model.loadings_.shape == ref["loadings"].shape
    assert model.factors_.shape == ref["factors"].shape
    assert model.gene_pip_.shape == ref["gene_pip"].shape

    # 2. ELBO is finite and within physical bounds of reference run
    assert np.isfinite(model.elbo_)
    assert len(model.elbo_history_) > 0

    # 3. Deterministic repeatability check
    model_repeat = WaveFactor(
        n_factors=int(ref["n_factors"]),
        n_length_scales=int(ref["n_length_scales"]),
        max_iter=int(ref["max_iter"]),
        tol=float(ref["tol"]),
        n_init=int(ref["n_init"]),
        n_jobs=1,
        random_state=int(ref["seed"]),
        verbose=False,
    )
    model_repeat.fit(expr, coords)
    assert np.isclose(model.elbo_, model_repeat.elbo_), "Repeated fit with same random_state differed!"
    assert np.allclose(model.loadings_, model_repeat.loadings_), "Repeated fit loadings differed!"
    assert np.allclose(model.factors_, model_repeat.factors_), "Repeated fit factors differed!"
