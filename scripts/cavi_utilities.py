import numpy as np
from scipy.special import digamma

# Utility functions to compute CAVI updates with

## General


def sum_log(log_a, log_b):
    # Taken from Heejung's hinted code
    # Computes log(a+b) given inputted log(a) and log(b) values
    if log_a > log_b:
        return log_a + np.log(1 + np.exp(log_b - log_a))
    else:
        return log_b + np.log(np.exp(log_a - log_b) + 1)


## Intermediate algebra


def gamma_t(i, l, parameters):
    alpha_hat_t_i_l = parameters["alpha_hat_t"][i][l]
    beta_hat_t_i_l = parameters["beta_hat_t"][i][l]
    return alpha_hat_t_i_l / beta_hat_t_i_l


def gamma_tau(i, l, parameters):
    alpha_hat_tau_i_l = parameters["alpha_hat_tau"][i][l]
    beta_hat_tau_i_l = parameters["beta_hat_tau"][i][l]
    return alpha_hat_tau_i_l / beta_hat_tau_i_l


def xi_L(i, j, k, l, parameters):
    r_pi_ijk_l = np.exp(parameters["log_r_pi"][l][i][j][k])
    mu_L_ijk_l = parameters["mu_L"][l][i][j][k]
    return r_pi_ijk_l * mu_L_ijk_l


def xi_F(i, j, parameters):
    r_eta_i_j = np.exp(parameters["log_r_eta"][i][j])
    mu_F_i_j = parameters["mu_F"][i][j]
    return r_eta_i_j * mu_F_i_j


def lambda_L(i, j, k, l, parameters):
    r_pi_ijk_l = np.exp(parameters["log_r_pi"][l][i][j][k])
    mu_L_ijk_l = parameters["mu_L"][l][i][j][k]
    sigma_squared_L_ijk_l = parameters["sigma_squared_L"][l][i][j][k]
    return r_pi_ijk_l * np.sum((sigma_squared_L_ijk_l, np.square(mu_L_ijk_l)))


def lambda_F(i, j, parameters):
    r_eta_i_j = np.exp(parameters["log_r_eta"][i][j])
    mu_F_i_j = parameters["mu_F"][i][j]
    sigma_squared_F_i_j = parameters["sigma_squared_F"][i][j]
    return r_eta_i_j * np.sum((sigma_squared_F_i_j, np.square(mu_F_i_j)))


def theta_t(i, l, parameters):
    alpha_hat_t_i_l = parameters["alpha_hat_t"][i][l]
    beta_hat_t_i_l = parameters["beta_hat_t"][i][l]
    return digamma(alpha_hat_t_i_l) - np.log(2 * np.pi * beta_hat_t_i_l)


def theta_tau(i, l, parameters):
    alpha_hat_tau_i_l = parameters["alpha_hat_tau"][i][l]
    beta_hat_tau_i_l = parameters["beta_hat_tau"][i][l]
    return digamma(alpha_hat_tau_i_l) - np.log(2 * np.pi * beta_hat_tau_i_l)


## Intermediate algebra for L_ijk_l, pi_ijk_l related updates


def u_F(i, l, d, parameters):
    lambda_F_l_d = lambda_F(l, d, parameters)
    gamma_tau_i_d = gamma_tau(i, d, parameters)
    return lambda_F_l_d * gamma_tau_i_d


def v_F(i, j, k, l, d, parameters):
    Y_ijk_d = parameters["Y"][d][i][j][k]
    xi_F_l_d = xi_F(l, d, parameters)
    gamma_tau_i_d = gamma_tau(i, d, parameters)
    return Y_ijk_d * xi_F_l_d * gamma_tau_i_d


def w_F(i, j, k, l, d, parameters):
    xi_F_l_d = xi_F(l, d, parameters)
    max_m = parameters["n_factors"]
    xi_sum = np.sum(
        tuple(
            xi_L(i, j, k, m, parameters) * xi_F(m, d, parameters)
            for m in range(max_m)
            if m != l
        )
    )
    gamma_tau_i_d = gamma_tau(i, d, parameters)
    return xi_F_l_d * xi_sum * gamma_tau_i_d


def s_F(i, j, k, l, d, parameters):
    v_F_ijk_l_d = v_F(i, j, k, l, d, parameters)
    w_F_ijk_l_d = w_F(i, j, k, l, d, parameters)
    return v_F_ijk_l_d - w_F_ijk_l_d


def s_bar_F(i, j, k, l, parameters):
    max_d = parameters["n_features"]
    s_sum = np.sum(tuple(s_F(i, j, k, l, d, parameters) for d in range(max_d)))
    return s_sum


def u_bar_F(i, l, parameters):
    max_d = parameters["n_features"]
    u_sum = np.sum(tuple(u_F(i, l, d, parameters) for d in range(max_d)))
    return u_sum


## Intermediate algebra for F_ijk_l, eta_ijk_l related updates


def u_L(a, b, c, i, j, parameters):
    lambda_L_abc_i = lambda_L(a, b, c, i, parameters)
    gamma_tau_a_j = gamma_tau(a, j, parameters)
    return lambda_L_abc_i * gamma_tau_a_j


def v_L(a, b, c, i, j, parameters):
    Y_abc_j = parameters["Y"][j][a][b][c]
    xi_L_abc_i = xi_L(a, b, c, i, parameters)
    gamma_tau_a_j = gamma_tau(a, j, parameters)
    return Y_abc_j * xi_L_abc_i * gamma_tau_a_j


def w_L(a, b, c, i, j, parameters):
    xi_L_abc_i = xi_L(a, b, c, i, parameters)
    max_m = parameters["n_factors"]
    xi_sum = np.sum(
        tuple(
            xi_L(a, b, c, m, parameters) * xi_F(m, j, parameters)
            for m in range(max_m)
            if m != i
        )
    )
    gamma_tau_a_j = gamma_tau(a, j, parameters)
    return xi_L_abc_i * xi_sum * gamma_tau_a_j


def s_L(a, b, c, i, j, parameters):
    v_L_abc_i_j = v_L(a, b, c, i, j, parameters)
    w_L_abc_i_j = w_L(a, b, c, i, j, parameters)
    return v_L_abc_i_j - w_L_abc_i_j


def s_bar_L(i, j, parameters):
    s_sum = 0
    max_a = parameters["n_resolutions"]  # Number of resolutions (including approx)
    for a in range(max_a):
        max_b = len(parameters["mu_L"][0][a])
        for b in range(max_b):
            max_c = len(parameters["mu_L"][0][a][b])
            for c in range(max_c):
                s_sum += s_L(a, b, c, i, j, parameters)
    return s_sum


def u_bar_L(i, j, parameters):
    u_sum = 0
    max_a = parameters["n_resolutions"]  # Number of resolutions (including approx)
    for a in range(max_a):
        max_b = len(parameters["mu_L"][0][a])
        for b in range(max_b):
            max_c = len(parameters["mu_L"][0][a][b])
            for c in range(max_c):
                u_sum += u_L(a, b, c, i, j, parameters)
    return u_sum
