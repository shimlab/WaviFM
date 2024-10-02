import numpy as np
from .cavi_utilities import *

# CAVI Updates

## Global constants
RELATIVE_PMF_INCREMENT = 1e-10  # Increment value used to ensure r_pi, r_eta updates computed will not be 0 nor 1 (so to avoid edge cases) that leads to problems such as ELBO computation.
LOG_RELATIVE_PMF_INCREMENT = np.log(RELATIVE_PMF_INCREMENT)

## For L_ijk_l, pi_ijk_l related updates


def compute_update_sigma_squared_L(i, j, k, l, parameters):
    gamma_t_i_l = gamma_t(i, l, parameters)
    u_bar_F_i_l = u_bar_F(i, l, parameters)
    return 1.0 / (gamma_t_i_l + u_bar_F_i_l)


def compute_update_mu_L(i, j, k, l, update_sigma_squared_L_ijk_l, parameters):
    s_bar_F_ijk_l = s_bar_F(i, j, k, l, parameters)
    return s_bar_F_ijk_l * update_sigma_squared_L_ijk_l


def compute_update_pi_ijk_l_log_relative_pmf(
    i, j, k, l, pi, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters
):
    # Preliminary computations
    theta_t_i_l = theta_t(i, l, parameters)
    log_p_pi_i = parameters["log_p_pi"][i]
    p_pi_i = np.exp(log_p_pi_i)

    # Computing factors to the relative log pdf
    log_scaling_factor = (
        0.5 * (np.log(2) + np.log(np.pi) + np.log(update_sigma_squared_L_ijk_l))
        if pi == 1
        else 0
    )
    log_exp_factor = (
        0.5
        * (theta_t_i_l + np.square(update_mu_L_ijk_l) / update_sigma_squared_L_ijk_l)
        if pi == 1
        else 0
    )
    log_bernoulli_factor = log_p_pi_i if pi == 1 else np.log(1 - p_pi_i)
    return log_scaling_factor + log_exp_factor + log_bernoulli_factor


def compute_update_log_r_pi(
    i, j, k, l, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters
):
    # Note: here doing incrementation after normalising so to hopefully combat issues of scale
    relative_true_log_prob = compute_update_pi_ijk_l_log_relative_pmf(
        i, j, k, l, 1, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters
    )
    relative_false_log_prob = compute_update_pi_ijk_l_log_relative_pmf(
        i, j, k, l, 0, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters
    )
    true_log_prob = relative_true_log_prob - sum_log(
        relative_true_log_prob, relative_false_log_prob
    )
    false_log_prob = relative_false_log_prob - sum_log(
        relative_true_log_prob, relative_false_log_prob
    )
    # Increment probabilities to avoid edge case 0 and 1 probabilities
    incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT)
    incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT)
    return incremented_true_log_prob - sum_log(
        incremented_true_log_prob, incremented_false_log_prob
    )


def compute_update_L_pi(i, j, k, l, parameters):
    update_sigma_squared_L_ijk_l = compute_update_sigma_squared_L(
        i, j, k, l, parameters
    )
    update_mu_L_ijk_l = compute_update_mu_L(
        i, j, k, l, update_sigma_squared_L_ijk_l, parameters
    )
    update_log_r_pi_ijk_l = compute_update_log_r_pi(
        i, j, k, l, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters
    )
    return {
        "sigma_squared_L": update_sigma_squared_L_ijk_l,
        "mu_L": update_mu_L_ijk_l,
        "log_r_pi": update_log_r_pi_ijk_l,
    }


def compute_update_sigma_squared_F(i, j, parameters):
    u_bar_L_i_j = u_bar_L(i, j, parameters)
    return 1.0 / (1.0 + u_bar_L_i_j)


def compute_update_mu_F(i, j, update_sigma_squared_F_i_j, parameters):
    s_bar_L_i_j = s_bar_L(i, j, parameters)
    return s_bar_L_i_j * update_sigma_squared_F_i_j


def compute_update_eta_i_j_log_relative_pmf(
    i, j, eta, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters
):
    log_p_eta_i = parameters["log_p_eta"][i]
    p_eta_i = np.exp(log_p_eta_i)

    # Computing factors to the relative pdf
    log_scaling_factor = (
        0.5 * (np.log(2) + np.log(np.pi) + np.log(update_sigma_squared_F_i_j))
        if eta == 1
        else 0
    )
    log_exp_factor = (
        0.5
        * (-np.log(2 * np.pi) + np.square(update_mu_F_i_j) / update_sigma_squared_F_i_j)
        if eta == 1
        else 0
    )
    log_bernoulli_factor = log_p_eta_i if eta == 1 else np.log(1 - p_eta_i)
    return log_scaling_factor + log_exp_factor + log_bernoulli_factor


def compute_update_log_r_eta(
    i, j, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters
):
    relative_true_log_prob = compute_update_eta_i_j_log_relative_pmf(
        i, j, 1, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters
    )
    relative_false_log_prob = compute_update_eta_i_j_log_relative_pmf(
        i, j, 0, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters
    )
    true_log_prob = relative_true_log_prob - sum_log(
        relative_true_log_prob, relative_false_log_prob
    )
    false_log_prob = relative_false_log_prob - sum_log(
        relative_true_log_prob, relative_false_log_prob
    )
    # Increment probabilities to avoid edge case 0 and 1 probabilities
    incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT)
    incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT)
    return incremented_true_log_prob - sum_log(
        incremented_true_log_prob, incremented_false_log_prob
    )


def compute_update_F_eta(i, j, parameters):
    update_sigma_squared_F_i_j = compute_update_sigma_squared_F(i, j, parameters)
    update_mu_F_i_j = compute_update_mu_F(i, j, update_sigma_squared_F_i_j, parameters)
    update_log_r_eta_i_j = compute_update_log_r_eta(
        i, j, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters
    )
    return {
        "sigma_squared_F": update_sigma_squared_F_i_j,
        "mu_F": update_mu_F_i_j,
        "log_r_eta": update_log_r_eta_i_j,
    }


## For tau_i_l related updates


def compute_update_alpha_hat_tau(i, l, parameters):
    alpha_tau_i_l = parameters["alpha_tau"][i][l]
    N_i = np.sum(
        np.fromiter(
            tuple(
                len(parameters["Y"][l][i][j]) for j in range(len(parameters["Y"][l][i]))
            ),
            dtype=np.int64,
        )
    )  # should be l independent in general (here int64 assumes a maximum of order of magnitude of 1 billion times 1 billion coefficients in every such coefficient matrix)
    return N_i / 2 + alpha_tau_i_l


def compute_update_beta_hat_tau(i, l, parameters):
    beta_tau_i_l = parameters["beta_tau"][i][l]

    # Decided to do separate loops for the individual terms below to help readability and debugging

    Y_squared_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            Y_ijk_l = parameters["Y"][l][i][j][k]
            Y_squared_sum += np.square(Y_ijk_l)

    Y_xi_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            Y_ijk_l = parameters["Y"][l][i][j][k]
            max_m = parameters["n_factors"]
            for m in range(max_m):
                Y_xi_sum += (
                    Y_ijk_l * xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters)
                )

    lambda_xi_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            max_m = parameters["n_factors"]
            for m in range(max_m):
                lambda_product = lambda_L(i, j, k, m, parameters) * lambda_F(
                    m, l, parameters
                )
                xi_product = xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters)
                lambda_xi_sum += lambda_product - np.square(xi_product)

    xi_product_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            max_m = parameters["n_factors"]
            xi_product_sum_for_m = 0
            for m in range(max_m):
                xi_product_sum_for_m += xi_L(i, j, k, m, parameters) * xi_F(
                    m, l, parameters
                )
            xi_product_sum += np.square(xi_product_sum_for_m)

    return beta_tau_i_l + 1 / 2 * (
        Y_squared_sum - 2 * Y_xi_sum + lambda_xi_sum + xi_product_sum
    )


def compute_update_tau(i, l, parameters):
    return {
        "alpha_hat_tau": compute_update_alpha_hat_tau(i, l, parameters),
        "beta_hat_tau": compute_update_beta_hat_tau(i, l, parameters),
    }


## For t_i_l related updates


def compute_update_alpha_hat_t(i, l, parameters):
    alpha_t_i_l = parameters["alpha_t"][i][l]

    r_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            r_sum += np.exp(parameters["log_r_pi"][l][i][j][k])

    return r_sum / 2 + alpha_t_i_l


def compute_update_beta_hat_t(i, l, parameters):
    beta_t_i_l = parameters["beta_t"][i][l]

    lambda_sum = 0
    max_j = len(parameters["Y"][l][i])
    for j in range(max_j):
        max_k = len(parameters["Y"][l][i][j])
        for k in range(max_k):
            lambda_sum += lambda_L(i, j, k, l, parameters)
    return beta_t_i_l + 1 / 2 * (lambda_sum)


def compute_update_t(i, l, parameters):
    return {
        "alpha_hat_t": compute_update_alpha_hat_t(i, l, parameters),
        "beta_hat_t": compute_update_beta_hat_t(i, l, parameters),
    }
