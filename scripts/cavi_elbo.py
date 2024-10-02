import numpy as np
from .cavi_utilities import *
from scipy.special import digamma, loggamma


def compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(i, j, k, l, parameters):
    theta_tau_i_l = theta_tau(i, l, parameters)
    gamma_tau_i_l = gamma_tau(i, l, parameters)
    Y_ijk_l = parameters["Y"][l][i][j][k]

    max_m = parameters["n_factors"]

    xi_product_sum = 0
    for m in range(max_m):
        xi_product_sum += xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters)

    xi_quad_product_sum = 0
    for a in range(max_m):
        for b in range(max_m):
            if a != b:
                xi_quad_product_sum += (
                    xi_L(i, j, k, a, parameters)
                    * xi_F(a, l, parameters)
                    * xi_L(i, j, k, b, parameters)
                    * xi_F(b, l, parameters)
                )

    lambda_product_sum = 0
    for m in range(max_m):
        lambda_product_sum += lambda_L(i, j, k, m, parameters) * lambda_F(
            m, l, parameters
        )

    return 0.5 * (
        theta_tau_i_l
        - gamma_tau_i_l
        * (
            np.square(Y_ijk_l)
            - 2 * Y_ijk_l * xi_product_sum
            + xi_quad_product_sum
            + lambda_product_sum
        )
    )


def compute_E_log_likelihood_L_ijk_l_given_pi_t(i, j, k, l, parameters):
    r_pi_ijk_l = np.exp(parameters["log_r_pi"][l][i][j][k])
    theta_t_i_l = theta_t(i, l, parameters)
    gamma_t_i_l = gamma_t(i, l, parameters)
    lambda_L_ijk_l = lambda_L(i, j, k, l, parameters)
    # Note the lingering expectation term was removed from below ELBO term as it'd cancel with those from the variational log likelihood expectation when computing ELBO (see corresponding obsidian derivations for details)
    return 0.5 * (r_pi_ijk_l * theta_t_i_l - gamma_t_i_l * lambda_L_ijk_l)


def compute_E_log_likelihood_F_i_j_given_eta(i, j, parameters):
    r_eta_i_j = np.exp(parameters["log_r_eta"][i][j])
    lambda_F_i_j = lambda_F(i, j, parameters)
    # Note the lingering expectation term was removed from below ELBO term as it'd cancel with those from the variational log likelihood expectation when computing ELBO (see corresponding obsidian derivations for details)
    return -0.5 * (r_eta_i_j * np.log(2 * np.pi) + lambda_F_i_j)


def compute_E_log_likelihood_pi_ijk_l(i, j, k, l, parameters):
    r_pi_ijk_l = np.exp(parameters["log_r_pi"][l][i][j][k])
    log_p_pi_i = parameters["log_p_pi"][i]
    p_pi_i = np.exp(log_p_pi_i)
    return r_pi_ijk_l * log_p_pi_i + (1 - r_pi_ijk_l) * np.log(1 - p_pi_i)


def compute_E_log_likelihood_eta_i_j(i, j, parameters):
    r_eta_i_j = np.exp(parameters["log_r_eta"][i][j])
    log_p_eta_i = parameters["log_p_eta"][i]
    p_eta_i = np.exp(log_p_eta_i)
    return r_eta_i_j * log_p_eta_i + (1 - r_eta_i_j) * np.log(1 - p_eta_i)


def compute_E_log_likelihood_t_i_l(i, l, parameters):
    alpha_t_i_l = parameters["alpha_t"][i][l]
    beta_t_i_l = parameters["beta_t"][i][l]
    alpha_hat_t_i_l = parameters["alpha_hat_t"][i][l]
    beta_hat_t_i_l = parameters["beta_hat_t"][i][l]
    gamma_t_i_l = gamma_t(i, l, parameters)
    return (
        (alpha_t_i_l - 1) * (digamma(alpha_hat_t_i_l) - np.log(beta_hat_t_i_l))
        - gamma_t_i_l * beta_t_i_l
        + alpha_t_i_l * np.log(beta_t_i_l)
        - loggamma(alpha_t_i_l)
    )


def compute_E_log_likelihood_tau_i_l(i, l, parameters):
    alpha_tau_i_l = parameters["alpha_tau"][i][l]
    beta_tau_i_l = parameters["beta_tau"][i][l]
    alpha_hat_tau_i_l = parameters["alpha_hat_tau"][i][l]
    beta_hat_tau_i_l = parameters["beta_hat_tau"][i][l]
    gamma_tau_i_l = gamma_tau(i, l, parameters)
    return (
        (alpha_tau_i_l - 1) * (digamma(alpha_hat_tau_i_l) - np.log(beta_hat_tau_i_l))
        - gamma_tau_i_l * beta_tau_i_l
        + alpha_tau_i_l * np.log(beta_tau_i_l)
        - loggamma(alpha_tau_i_l)
    )


def compute_E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l(
    i, j, k, l, parameters
):
    r_pi_ijk_l = np.exp(parameters["log_r_pi"][l][i][j][k])
    sigma_squared_L_ijk_l = parameters["sigma_squared_L"][l][i][j][k]
    # Note the lingering expectation term was removed from below ELBO term as it'd cancel with those from the log likelihood expectation when computing ELBO (see corresponding obsidian derivations for details)
    return (
        (r_pi_ijk_l / 2) * np.log(2 * np.pi * sigma_squared_L_ijk_l + 1)
        - r_pi_ijk_l * np.log(r_pi_ijk_l)
        - (1 - r_pi_ijk_l) * np.log(1 - r_pi_ijk_l)
    )


def compute_E_negative_variational_log_likelihood_F_i_j_eta_i_j(i, j, parameters):
    r_eta_i_j = np.exp(parameters["log_r_eta"][i][j])
    sigma_squared_F_i_j = parameters["sigma_squared_F"][i][j]
    # Note the lingering expectation term was removed from below ELBO term as it'd cancel with those from the log likelihood expectation when computing ELBO (see corresponding obsidian derivations for details)
    return (
        (r_eta_i_j / 2) * np.log(2 * np.pi * sigma_squared_F_i_j + 1)
        - r_eta_i_j * np.log(r_eta_i_j)
        - (1 - r_eta_i_j) * np.log(1 - r_eta_i_j)
    )


def compute_E_negative_variational_log_likelihood_t_i_l(i, l, parameters):
    alpha_hat_t_i_l = parameters["alpha_hat_t"][i][l]
    beta_hat_t_i_l = parameters["beta_hat_t"][i][l]
    return (
        alpha_hat_t_i_l
        - np.log(beta_hat_t_i_l)
        + loggamma(alpha_hat_t_i_l)
        + (1 - alpha_hat_t_i_l) * digamma(alpha_hat_t_i_l)
    )


def compute_E_negative_variational_log_likelihood_tau_i_l(i, l, parameters):
    alpha_hat_tau_i_l = parameters["alpha_hat_tau"][i][l]
    beta_hat_tau_i_l = parameters["beta_hat_tau"][i][l]
    return (
        alpha_hat_tau_i_l
        - np.log(beta_hat_tau_i_l)
        + loggamma(alpha_hat_tau_i_l)
        + (1 - alpha_hat_tau_i_l) * digamma(alpha_hat_tau_i_l)
    )


def compute_elbo(parameters):
    elbo = 0

    # Summing terms across factors and features
    n_factors = parameters["n_factors"]
    n_features = parameters["n_features"]

    # Summing terms across factors and features
    for i in range(n_factors):
        for j in range(n_features):
            elbo += compute_E_log_likelihood_F_i_j_given_eta(i, j, parameters)
            elbo += compute_E_log_likelihood_eta_i_j(i, j, parameters)
            elbo += compute_E_negative_variational_log_likelihood_F_i_j_eta_i_j(
                i, j, parameters
            )

    # Summing terms across wavelet resolution and factors
    for l in range(n_factors):
        for i in range(len(parameters["mu_L"][l])):
            elbo += compute_E_log_likelihood_t_i_l(i, l, parameters)
            elbo += compute_E_negative_variational_log_likelihood_t_i_l(
                i, l, parameters
            )

    # Summing terms across wavelet resolution and features
    for l in range(n_features):
        for i in range(len(parameters["Y"][l])):
            elbo += compute_E_log_likelihood_tau_i_l(i, l, parameters)
            elbo += compute_E_negative_variational_log_likelihood_tau_i_l(
                i, l, parameters
            )

    # Summing terms across wavelet coefficients and factors
    for l in range(n_factors):
        for i in range(len(parameters["mu_L"][l])):
            for j in range(len(parameters["mu_L"][l][i])):
                for k in range(len(parameters["mu_L"][l][i][j])):
                    elbo += compute_E_log_likelihood_L_ijk_l_given_pi_t(
                        i, j, k, l, parameters
                    )
                    elbo += compute_E_log_likelihood_pi_ijk_l(i, j, k, l, parameters)
                    elbo += (
                        compute_E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l(
                            i, j, k, l, parameters
                        )
                    )

    # Summing terms across wavelet coefficients and features
    x = 0
    for l in range(n_features):
        for i in range(len(parameters["Y"][l])):
            for j in range(len(parameters["Y"][l][i])):
                for k in range(len(parameters["Y"][l][i][j])):
                    x += 1
                    elbo += compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(
                        i, j, k, l, parameters
                    )
    return elbo
