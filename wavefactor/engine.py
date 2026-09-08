"""
Coordinate Ascent Variational Inference (CAVI) execution engine for WaveFactor.
Interactions with the C++ Pybind11 backend and multiprocessing parallelization.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import time
import copy
import numpy as np
from multiprocessing import Pool

# Attempt to load compiled C++ backend
_CPP_MODULE = None

def _get_cpp_backend():
    """Dynamically locates and loads the compiled C++ WaveFactor Pybind11 module."""
    global _CPP_MODULE
    if _CPP_MODULE is not None:
        return _CPP_MODULE

    # Try standard import names
    import_candidates = [
        "WaveFactor",
        "wavefactor._cpp",
        "build.WaveFactor",
    ]

    for cand in import_candidates:
        try:
            mod = __import__(cand, fromlist=["cavi", "Parameters", "CaviDimensions", "CaviResult"])
            if hasattr(mod, "Parameters"):
                _CPP_MODULE = mod
                return _CPP_MODULE
        except (ImportError, ModuleNotFoundError):
            pass

    # Check build directories relative to current file (including MSVC Release/Debug directories)
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_dirs = [
        os.path.join(pkg_dir, "build"),
        os.path.join(pkg_dir, "build", "Release"),
        os.path.join(pkg_dir, "build", "Debug"),
        os.path.join(pkg_dir, "build", "src"),
    ]
    for cand_dir in candidate_dirs:
        if os.path.exists(cand_dir) and cand_dir not in sys.path:
            sys.path.insert(0, cand_dir)
            try:
                import WaveFactor as mod
                if hasattr(mod, "Parameters"):
                    _CPP_MODULE = mod
                    return _CPP_MODULE
            except (ImportError, ModuleNotFoundError):
                pass

    return None


def init_parameters(
    Y: List,
    dimensions: Dict[str, Any],
    priors: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    """
    Initializes CAVI variational parameters and model prior hyperparameters.
    """
    if rng is None:
        rng = np.random.mtrand._rand

    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    p_pi_shape = dimensions["p_pi_shape"]
    F_shape = dimensions["F_shape"]
    ab_t_shape = dimensions["ab_t_shape"]
    ab_tau_shape = dimensions["ab_tau_shape"]
    L_shape = dimensions["L_shape"]

    # Initialise variational parameters matching exact pre-v2.0 RNG draw sequence
    def _draw_L_shaped(low, high, log_transform=False):
        vals = []
        for l in range(n_factors):
            f_vals = []
            for i in range(n_resolutions):
                lvl_vals = []
                for j in range(len(L_shape[l][i])):
                    matrix_shape = L_shape[l][i][j].shape
                    draw = rng.uniform(low, high, size=matrix_shape)
                    lvl_vals.append(np.log(draw) if log_transform else draw)
                f_vals.append(lvl_vals)
            vals.append(f_vals)
        return vals

    mu_L = _draw_L_shaped(-10.0, 10.0, log_transform=False)
    sigma_squared_L = _draw_L_shaped(0.01, 10.0, log_transform=False)
    log_r_pi = _draw_L_shaped(0.01, 0.99, log_transform=True)

    mu_F = (rng.rand(*F_shape) * 20.0 - 10.0).astype(np.float64)
    sigma_squared_F = (rng.rand(*F_shape) * 9.99 + 0.01).astype(np.float64)
    r_eta = (rng.rand(*F_shape) * 0.98 + 0.01).astype(np.float64)
    log_r_eta = np.log(r_eta)
    alpha_hat_t = (rng.rand(*ab_t_shape) * 9.99 + 0.01).astype(np.float64)
    beta_hat_t = (rng.rand(*ab_t_shape) * 9.99 + 0.01).astype(np.float64)
    alpha_hat_tau = (rng.rand(*ab_tau_shape) * 9.99 + 0.01).astype(np.float64)
    beta_hat_tau = (rng.rand(*ab_tau_shape) * 9.99 + 0.01).astype(np.float64)

    # Set prior hyperparameters
    default_log_p_pi = np.log(np.full(p_pi_shape, 0.5, dtype=np.float64))
    default_log_p_eta = np.log(np.full(F_shape, 0.5, dtype=np.float64))
    default_alpha_t = np.full(ab_t_shape, 1.0, dtype=np.float64)
    default_beta_t = np.full(ab_t_shape, 1.0, dtype=np.float64)
    default_alpha_tau = np.full(ab_tau_shape, 1.0, dtype=np.float64)
    default_beta_tau = np.full(ab_tau_shape, 1.0, dtype=np.float64)

    if priors:
        log_p_pi = priors.get("log_p_pi", default_log_p_pi)
        log_p_eta = priors.get("log_p_eta", default_log_p_eta)
        alpha_t = priors.get("alpha_t", default_alpha_t)
        beta_t = priors.get("beta_t", default_beta_t)
        alpha_tau = priors.get("alpha_tau", default_alpha_tau)
        beta_tau = priors.get("beta_tau", default_beta_tau)
    else:
        log_p_pi = default_log_p_pi
        log_p_eta = default_log_p_eta
        alpha_t = default_alpha_t
        beta_t = default_beta_t
        alpha_tau = default_alpha_tau
        beta_tau = default_beta_tau

    return {
        "n_resolutions": n_resolutions,
        "n_factors": n_factors,
        "n_features": n_features,
        "Y": Y,
        "log_p_pi": log_p_pi,
        "log_p_eta": log_p_eta,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "alpha_tau": alpha_tau,
        "beta_tau": beta_tau,
        "mu_L": mu_L,
        "sigma_squared_L": sigma_squared_L,
        "log_r_pi": log_r_pi,
        "mu_F": mu_F,
        "sigma_squared_F": sigma_squared_F,
        "log_r_eta": log_r_eta,
        "alpha_hat_t": alpha_hat_t,
        "beta_hat_t": beta_hat_t,
        "alpha_hat_tau": alpha_hat_tau,
        "beta_hat_tau": beta_hat_tau,
    }


def build_parameters_cpp(parameters: Dict[str, Any], cpp_backend: Any) -> Any:
    """Instantiates C++ Parameters class object from Python parameters dictionary."""
    return cpp_backend.Parameters(
        parameters["n_resolutions"],
        parameters["n_factors"],
        parameters["n_features"],
        parameters["Y"],
        parameters["log_p_pi"].tolist() if isinstance(parameters["log_p_pi"], np.ndarray) else parameters["log_p_pi"],
        parameters["log_p_eta"].tolist() if isinstance(parameters["log_p_eta"], np.ndarray) else parameters["log_p_eta"],
        parameters["alpha_t"].tolist() if isinstance(parameters["alpha_t"], np.ndarray) else parameters["alpha_t"],
        parameters["beta_t"].tolist() if isinstance(parameters["beta_t"], np.ndarray) else parameters["beta_t"],
        parameters["alpha_tau"].tolist() if isinstance(parameters["alpha_tau"], np.ndarray) else parameters["alpha_tau"],
        parameters["beta_tau"].tolist() if isinstance(parameters["beta_tau"], np.ndarray) else parameters["beta_tau"],
        parameters["mu_L"],
        parameters["sigma_squared_L"],
        parameters["log_r_pi"],
        parameters["mu_F"].tolist() if isinstance(parameters["mu_F"], np.ndarray) else parameters["mu_F"],
        parameters["sigma_squared_F"].tolist() if isinstance(parameters["sigma_squared_F"], np.ndarray) else parameters["sigma_squared_F"],
        parameters["log_r_eta"].tolist() if isinstance(parameters["log_r_eta"], np.ndarray) else parameters["log_r_eta"],
        parameters["alpha_hat_t"].tolist() if isinstance(parameters["alpha_hat_t"], np.ndarray) else parameters["alpha_hat_t"],
        parameters["beta_hat_t"].tolist() if isinstance(parameters["beta_hat_t"], np.ndarray) else parameters["beta_hat_t"],
        parameters["alpha_hat_tau"].tolist() if isinstance(parameters["alpha_hat_tau"], np.ndarray) else parameters["alpha_hat_tau"],
        parameters["beta_hat_tau"].tolist() if isinstance(parameters["beta_hat_tau"], np.ndarray) else parameters["beta_hat_tau"],
    )


def extract_cpp_parameters_to_dict(cpp_params: Any) -> Dict[str, Any]:
    """Converts a C++ Parameters Pybind11 object to a standard Python dictionary."""
    return {
        "n_resolutions": cpp_params.n_resolutions,
        "n_factors": cpp_params.n_factors,
        "n_features": cpp_params.n_features,
        "Y": cpp_params.Y,
        "log_p_pi": np.asarray(cpp_params.log_p_pi, dtype=np.float64),
        "log_p_eta": np.asarray(cpp_params.log_p_eta, dtype=np.float64),
        "alpha_t": np.asarray(cpp_params.alpha_t, dtype=np.float64),
        "beta_t": np.asarray(cpp_params.beta_t, dtype=np.float64),
        "alpha_tau": np.asarray(cpp_params.alpha_tau, dtype=np.float64),
        "beta_tau": np.asarray(cpp_params.beta_tau, dtype=np.float64),
        "mu_L": cpp_params.mu_L,
        "sigma_squared_L": cpp_params.sigma_squared_L,
        "log_r_pi": cpp_params.log_r_pi,
        "mu_F": np.asarray(cpp_params.mu_F, dtype=np.float64),
        "sigma_squared_F": np.asarray(cpp_params.sigma_squared_F, dtype=np.float64),
        "log_r_eta": np.asarray(cpp_params.log_r_eta, dtype=np.float64),
        "alpha_hat_t": np.asarray(cpp_params.alpha_hat_t, dtype=np.float64),
        "beta_hat_t": np.asarray(cpp_params.beta_hat_t, dtype=np.float64),
        "alpha_hat_tau": np.asarray(cpp_params.alpha_hat_tau, dtype=np.float64),
        "beta_hat_tau": np.asarray(cpp_params.beta_hat_tau, dtype=np.float64),
    }


def _run_single_cavi_worker(
    params_dict: Dict[str, Any],
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    """Worker task for multiprocessing execution of a single CAVI run."""
    cpp = _get_cpp_backend()
    if cpp is None:
        raise RuntimeError("C++ WaveFactor module is required for CAVI execution.")

    cpp_params = build_parameters_cpp(params_dict, cpp)
    t0 = time.time()
    result_cpp = cpp.cavi(cpp_params, max_iter, tol)
    t1 = time.time()

    return {
        "parameters": extract_cpp_parameters_to_dict(result_cpp.parameters),
        "elbo_record": list(result_cpp.elbo_record),
        "elbo": float(result_cpp.elbo),
        "cpp_time": t1 - t0,
    }


def run_cavi(
    true_Y: List,
    dimensions: Dict[str, Any],
    priors: Optional[Dict[str, Any]] = None,
    max_iter: int = 1000,
    tol: float = 1e-5,
    n_init: int = 5,
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrates multi-start Coordinate Ascent Variational Inference.

    Parameters
    ----------
    true_Y : list
        2D DWT coefficient matrices per feature.
    dimensions : dict
        Dimension specification dictionary.
    priors : dict, optional
        Prior hyperparameter matrices.
    max_iter : int, default=1000
        Maximum CAVI iterations per start.
    tol : float, default=1e-5
        Relative ELBO convergence tolerance.
    n_init : int, default=5
        Number of random initializations.
    n_jobs : int, default=-1
        Number of parallel worker processes (-1 uses all available CPU cores).
    random_state : int, optional
        Seed for deterministic random number generation.
    verbose : bool, default=True
        Whether to print progress messages.

    Returns
    -------
    dict
        Dictionary containing best parameters, elbo_record, elbo, and cpp_time.
    """
    cpp = _get_cpp_backend()
    if cpp is None:
        raise RuntimeError(
            "WaveFactor C++ backend not found. "
            "Please ensure the C++ extension module is compiled."
        )

    # Initialize master RNG for reproducible initializations across workers
    master_rng = np.random.RandomState(random_state)
    seeds = master_rng.randint(0, 2**31 - 1, size=n_init)

    # Generate initial parameter states
    params_list = [
        init_parameters(true_Y, dimensions, priors=priors, rng=np.random.RandomState(s))
        for s in seeds
    ]

    results_list = []

    # Run in parallel if n_jobs != 1 and n_init > 1
    if n_jobs != 1 and n_init > 1:
        num_workers = os.cpu_count() if n_jobs == -1 else max(1, n_jobs)
        num_workers = min(num_workers, n_init)
        tasks = [(p, max_iter, tol) for p in params_list]
        with Pool(processes=num_workers) as pool:
            results_list = pool.starmap(_run_single_cavi_worker, tasks)
    else:
        # Serial execution
        for i, p in enumerate(params_list):
            res = _run_single_cavi_worker(p, max_iter, tol)
            results_list.append(res)
            if verbose:
                n_iterations = len(res["elbo_record"]) - 1
                print(
                    f"Initialisation {i+1}:\n"
                    f"\tELBO = {res['elbo']}\n"
                    f"\t#Iterations = {n_iterations}\n"
                    f"\tTime taken (s) = {res['cpp_time']:.2f}"
                )

    if verbose and (n_jobs != 1 and n_init > 1):
        for i, res in enumerate(results_list):
            n_iterations = len(res["elbo_record"]) - 1
            print(
                f"Initialisation {i+1}:\n"
                f"\tELBO = {res['elbo']}\n"
                f"\t#Iterations = {n_iterations}\n"
                f"\tTime taken (s) = {res['cpp_time']:.2f}"
            )

    # Select initialization with maximal ELBO
    best_idx, best_result = max(enumerate(results_list), key=lambda item: item[1]["elbo"])
    if verbose:
        print(f"Initialisation {best_idx + 1} has maximal ELBO and is returned")

    return best_result
