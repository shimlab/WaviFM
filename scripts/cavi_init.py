# This file contains code to initialise CAVI algorithm

# Code to enable this notebook to import from libraries
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
from .utilities import *
import build.WaviFM as WaviFM


def init_parameters(Y, dimensions, priors=None):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    p_pi_shape = dimensions["p_pi_shape"]
    p_eta_shape = dimensions["p_eta_shape"]
    F_shape = dimensions["F_shape"]
    ab_t_shape = dimensions["ab_t_shape"]
    ab_tau_shape = dimensions["ab_tau_shape"]

    # Initialise algorithm
    mu_L = L_shaped_rand(-10, 10, dimensions)
    sigma_squared_L = L_shaped_rand(0.01, 10, dimensions)
    log_r_pi = L_shaped_log_rand(0.01, 0.99, dimensions)
    mu_F = (np.random.rand(*F_shape) * 20 - 10).astype(np.float64)
    sigma_squared_F = (np.random.rand(*F_shape) * 9.99 + 0.01).astype(np.float64)
    r_eta = (np.random.rand(*F_shape) * 0.98 + 0.01).astype(np.float64)
    log_r_eta = np.log(r_eta)
    alpha_hat_t = (np.random.rand(*ab_t_shape) * 9.99 + 0.01).astype(np.float64)
    beta_hat_t = (np.random.rand(*ab_t_shape) * 9.99 + 0.01).astype(np.float64)
    alpha_hat_tau = (np.random.rand(*ab_tau_shape) * 9.99 + 0.01).astype(np.float64)
    beta_hat_tau = (np.random.rand(*ab_tau_shape) * 9.99 + 0.01).astype(np.float64)
    
    # Set priors
    default_log_p_pi = np.log(np.full(p_pi_shape, 0.5).astype(np.float64))
    default_log_p_eta = np.log(np.full(p_eta_shape, 0.5).astype(np.float64))
    default_alpha_t = np.full(ab_t_shape, 1).astype(np.float64)
    default_beta_t = np.full(ab_t_shape, 1).astype(np.float64)
    default_alpha_tau = np.full(ab_tau_shape, 1).astype(np.float64)
    default_beta_tau = np.full(ab_tau_shape, 1).astype(np.float64)
    
    if priors:
        if "log_p_pi" in priors:
            log_p_pi = priors["log_p_pi"]
        else:
            log_p_pi = default_log_p_pi
        
        if "log_p_eta" in priors:
            log_p_eta = priors["log_p_eta"]
        else:
            log_p_eta = default_log_p_eta
        
        if "alpha_t" in priors:
            alpha_t = priors["alpha_t"]
        else:
            alpha_t = default_alpha_t
        
        if "beta_t" in priors:
            beta_t = priors["beta_t"]
        else:
            beta_t = default_beta_t
        
        if "alpha_tau" in priors:
            alpha_tau = priors["alpha_tau"]
        else:
            alpha_tau = default_alpha_tau
        
        if "beta_tau" in priors:
            beta_tau = priors["beta_tau"]
        else:
            beta_tau = default_beta_tau
    else:
        log_p_pi = default_log_p_pi
        log_p_eta = default_log_p_eta
        alpha_t = default_alpha_t
        beta_t = default_beta_t
        alpha_tau = default_alpha_tau
        beta_tau = default_beta_tau

    parameters = {
        "n_resolutions": n_resolutions,
        "n_factors": n_factors,
        "n_features": n_features,
        "Y": Y,  # This and below are model prior hyperparameters
        "log_p_pi": log_p_pi,
        "log_p_eta": log_p_eta,
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "alpha_tau": alpha_tau,
        "beta_tau": beta_tau,
        "mu_L": mu_L,  # This and below are variational hyperparameters
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

    return parameters


def init_parameters_cpp(Y, dimensions, priors):

    parameters = init_parameters(Y, dimensions, priors)

    # Get parameters input in the form accepted by the C++ cavi code
    parameters_cpp = WaviFM.Parameters(
        parameters["n_resolutions"],
        parameters["n_factors"],
        parameters["n_features"],
        parameters["Y"],
        parameters["log_p_pi"].tolist(),
        parameters["log_p_eta"].tolist(),
        parameters["alpha_t"].tolist(),
        parameters["beta_t"].tolist(),
        parameters["alpha_tau"].tolist(),
        parameters["beta_tau"].tolist(),
        parameters["mu_L"],
        parameters["sigma_squared_L"],
        parameters["log_r_pi"],
        parameters["mu_F"].tolist(),
        parameters["sigma_squared_F"].tolist(),
        parameters["log_r_eta"].tolist(),
        parameters["alpha_hat_t"].tolist(),
        parameters["beta_hat_t"].tolist(),
        parameters["alpha_hat_tau"].tolist(),
        parameters["beta_hat_tau"].tolist(),
    )

    return parameters_cpp
