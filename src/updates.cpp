#include "updates.hpp"

// Global constants
const double RELATIVE_PMF_INCREMENT = 1e-10;
const double LOG_RELATIVE_PMF_INCREMENT = std::log(RELATIVE_PMF_INCREMENT);

// For L_ijk_l, pi_ijk_l related updates
double compute_update_sigma_squared_L(int i, int j, int k, int l, const Parameters &parameters)
{
    double gamma_t_i_l = gamma_t(i, l, parameters);
    double u_bar_F_i_l = u_bar_F(i, l, parameters);
    return 1.0 / (gamma_t_i_l + u_bar_F_i_l);
}

double compute_update_mu_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, const Parameters &parameters)
{
    double s_bar_F_ijk_l = s_bar_F(i, j, k, l, parameters);
    return s_bar_F_ijk_l * update_sigma_squared_L_ijk_l;
}

double compute_update_pi_ijk_l_log_relative_pmf(int i, int j, int k, int l, int pi, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters)
{
    double theta_t_i_l = theta_t(i, l, parameters);
    double log_p_pi_i = parameters.log_p_pi[i];
    double p_pi_i = exp(log_p_pi_i);

    double log_scaling_factor = 0;
    double log_exp_factor = 0;
    double log_bernoulli_factor = (pi == 1) ? log_p_pi_i : log(1 - p_pi_i);
    if (pi == 1)
    {
        log_scaling_factor = 0.5 * (log(2) + log(M_PI) + log(update_sigma_squared_L_ijk_l));
        log_exp_factor = 0.5 * (theta_t_i_l + update_mu_L_ijk_l * update_mu_L_ijk_l / update_sigma_squared_L_ijk_l);
    }

    return log_scaling_factor + log_exp_factor + log_bernoulli_factor;
}

double compute_update_log_r_pi(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters)
{
    double relative_true_log_prob = compute_update_pi_ijk_l_log_relative_pmf(i, j, k, l, 1, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);
    double relative_false_log_prob = compute_update_pi_ijk_l_log_relative_pmf(i, j, k, l, 0, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);

    double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
    double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

    double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
    double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

    return incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);
}

UpdateLPIResult compute_update_L_pi(int i, int j, int k, int l, const Parameters &parameters)
{
    double update_sigma_squared_L_ijk_l = compute_update_sigma_squared_L(i, j, k, l, parameters);
    double update_mu_L_ijk_l = compute_update_mu_L(i, j, k, l, update_sigma_squared_L_ijk_l, parameters);
    double update_log_r_pi_ijk_l = compute_update_log_r_pi(i, j, k, l, update_sigma_squared_L_ijk_l, update_mu_L_ijk_l, parameters);

    return {
        update_sigma_squared_L_ijk_l,
        update_mu_L_ijk_l,
        update_log_r_pi_ijk_l};
}

// For F_i_j, eta_i_j related updates
double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters)
{
    double u_bar_L_i_j = u_bar_L(i, j, parameters);
    return 1.0 / (1 + u_bar_L_i_j);
}

double compute_update_mu_F(int i, int j, double update_sigma_squared_F_i_j, const Parameters &parameters)
{
    double s_bar_L_i_j = s_bar_L(i, j, parameters);
    return s_bar_L_i_j * update_sigma_squared_F_i_j;
}

double compute_update_eta_i_j_log_relative_pmf(int i, int j, int eta, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters)
{
    double log_p_eta_i = parameters.log_p_eta[i];
    double p_eta_i = exp(log_p_eta_i);

    double log_scaling_factor = 0;
    double log_exp_factor = 0;
    double log_bernoulli_factor = (eta == 1) ? log_p_eta_i : log(1 - p_eta_i);
    if (eta == 1)
    {
        log_scaling_factor = 0.5 * (log(2) + log(M_PI) + log(update_sigma_squared_F_i_j));
        log_exp_factor = 0.5 * (-log(2 * M_PI) + update_mu_F_i_j * update_mu_F_i_j / update_sigma_squared_F_i_j);
    }

    return log_scaling_factor + log_exp_factor + log_bernoulli_factor;
}

double compute_update_log_r_eta(int i, int j, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters)
{
    double relative_true_log_prob = compute_update_eta_i_j_log_relative_pmf(i, j, 1, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);
    double relative_false_log_prob = compute_update_eta_i_j_log_relative_pmf(i, j, 0, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);

    double true_log_prob = relative_true_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);
    double false_log_prob = relative_false_log_prob - sum_log(relative_true_log_prob, relative_false_log_prob);

    double incremented_true_log_prob = sum_log(true_log_prob, LOG_RELATIVE_PMF_INCREMENT);
    double incremented_false_log_prob = sum_log(false_log_prob, LOG_RELATIVE_PMF_INCREMENT);

    return incremented_true_log_prob - sum_log(incremented_true_log_prob, incremented_false_log_prob);
}

UpdateFEtaResult compute_update_F_eta(int i, int j, const Parameters &parameters)
{
    double update_sigma_squared_F_i_j = compute_update_sigma_squared_F(i, j, parameters);
    double update_mu_F_i_j = compute_update_mu_F(i, j, update_sigma_squared_F_i_j, parameters);
    double update_log_r_eta_i_j = compute_update_log_r_eta(i, j, update_sigma_squared_F_i_j, update_mu_F_i_j, parameters);

    return {
        update_sigma_squared_F_i_j,
        update_mu_F_i_j,
        update_log_r_eta_i_j};
}

// For tau_i_l related updates
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters)
{
    double alpha_tau_i_l = parameters.alpha_tau[i][l];
    double N_i = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        N_i += parameters.Y[l][i][j].size();
    }

    return N_i / 2.0 + alpha_tau_i_l;
}

double compute_update_beta_hat_tau(int i, int l, const Parameters &parameters)
{
    double beta_tau_i_l = parameters.beta_tau[i][l];

    // Calculate Y_squared_sum
    double Y_squared_sum = 0.0;
    size_t max_j = parameters.Y[l][i].size();
    size_t max_m = parameters.n_factors;
    for (size_t j = 0; j < max_j; ++j)
    {
        size_t max_k = parameters.Y[l][i][j].size();
        for (size_t k = 0; k < max_k; ++k)
        {
            double Y_ijk_l = parameters.Y[l][i][j][k];
            Y_squared_sum += Y_ijk_l * Y_ijk_l;
        }
    }

    // Calculate Y_xi_sum
    double Y_xi_sum = 0.0;
    for (size_t j = 0; j < max_j; ++j)
    {
        size_t max_k = parameters.Y[l][i][j].size();
        for (size_t k = 0; k < max_k; ++k)
        {
            double Y_ijk_l = parameters.Y[l][i][j][k];
            for (size_t m = 0; m < max_m; ++m)
            {
                Y_xi_sum += Y_ijk_l * xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters);
            }
        }
    }

    // Calculate lambda_xi_sum
    double lambda_xi_sum = 0.0;
    for (size_t j = 0; j < max_j; ++j)
    {
        size_t max_k = parameters.Y[l][i][j].size();
        for (size_t k = 0; k < max_k; ++k)
        {
            for (size_t m = 0; m < max_m; ++m)
            {
                double lambda_product = lambda_L(i, j, k, m, parameters) * lambda_F(m, l, parameters);
                double xi_product = xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters);
                lambda_xi_sum += lambda_product - xi_product * xi_product;
            }
        }
    }

    // Calculate xi_product_sum
    double xi_product_sum = 0.0;
    for (size_t j = 0; j < max_j; ++j)
    {
        size_t max_k = parameters.Y[l][i][j].size();
        for (size_t k = 0; k < max_k; ++k)
        {
            double xi_product_sum_for_m = 0.0;
            for (size_t m = 0; m < max_m; ++m)
            {
                xi_product_sum_for_m += xi_L(i, j, k, m, parameters) * xi_F(m, l, parameters);
            }
            xi_product_sum += xi_product_sum_for_m * xi_product_sum_for_m;
        }
    }

    return beta_tau_i_l + 0.5 * (Y_squared_sum - 2 * Y_xi_sum + lambda_xi_sum + xi_product_sum);
}

UpdateTauResult compute_update_tau(int i, int l, const Parameters &parameters)
{
    double alpha_hat_tau = compute_update_alpha_hat_tau(i, l, parameters);
    double beta_hat_tau = compute_update_beta_hat_tau(i, l, parameters);

    return {
        alpha_hat_tau,
        beta_hat_tau};
}

double compute_update_alpha_hat_t(int i, int l, const Parameters &parameters)
{
    double alpha_t_i_l = parameters.alpha_t[i][l];

    double r_sum = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        for (size_t k = 0; k < parameters.Y[l][i][j].size(); ++k)
        {
            r_sum += std::exp(parameters.log_r_pi[l][i][j][k]);
        }
    }

    return r_sum / 2.0 + alpha_t_i_l;
}

double compute_update_beta_hat_t(int i, int l, const Parameters &parameters)
{
    double beta_t_i_l = parameters.beta_t[i][l];

    double lambda_sum = 0.0;
    for (size_t j = 0; j < parameters.Y[l][i].size(); ++j)
    {
        for (size_t k = 0; k < parameters.Y[l][i][j].size(); ++k)
        {
            lambda_sum += lambda_L(i, j, k, l, parameters);
        }
    }
    return beta_t_i_l + 0.5 * lambda_sum;
}

UpdateTResult compute_update_t(int i, int l, const Parameters &parameters)
{
    double alpha_hat_t = compute_update_alpha_hat_t(i, l, parameters);
    double beta_hat_t = compute_update_beta_hat_t(i, l, parameters);

    return {
        alpha_hat_t,
        beta_hat_t};
}