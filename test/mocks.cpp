#include "mocks.hpp"

// Mock Parameters() instance, with dimension parameters given below, log(0.5) as the entry values of all probability parameters (log_p_pi, log_p_eta, log_r_pi, log_r_eta), and 1 for entries of all other parameters.
namespace mocks
{
    // Mock constants
    double log_half = std::log(0.5);

    // Mock dimension parameters
    int n_factors = 2;
    int n_resolutions = 2;
    int n_features = 3;

    // Mock tensor parameters
    Tensor4D Y = {
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}},
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}},
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}}};

    Tensor1D log_p_pi = {log_half, log_half};

    Tensor1D log_p_eta = {log_half, log_half};

    Tensor2D alpha_t = {{1, 1},
                        {1, 1}};

    Tensor2D beta_t = {{1, 1},
                       {1, 1}};

    Tensor2D alpha_tau = {{1, 1, 1},
                          {1, 1, 1}};

    Tensor2D beta_tau = {{1, 1, 1},
                         {1, 1, 1}};

    Tensor4D mu_L = {
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}},
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}}};

    Tensor4D sigma_squared_L = {
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}},
        {{{1, 1, 1, 1}},
         {{1, 1, 1, 1},
          {1, 1, 1, 1},
          {1, 1, 1, 1}}}};

    Tensor4D log_r_pi = {
        {{{log_half, log_half, log_half, log_half}},
         {{log_half, log_half, log_half, log_half},
          {log_half, log_half, log_half, log_half},
          {log_half, log_half, log_half, log_half}}},
        {{{log_half, log_half, log_half, log_half}},
         {{log_half, log_half, log_half, log_half},
          {log_half, log_half, log_half, log_half},
          {log_half, log_half, log_half, log_half}}}};

    Tensor2D mu_F = {{1, 1, 1},
                     {1, 1, 1}};

    Tensor2D sigma_squared_F = {{1, 1, 1},
                                {1, 1, 1}};

    Tensor2D log_r_eta = {{log_half, log_half, log_half},
                          {log_half, log_half, log_half}};

    Tensor2D alpha_hat_t = {{1, 1},
                            {1, 1}};

    Tensor2D beta_hat_t = {{1, 1},
                           {1, 1}};

    Tensor2D alpha_hat_tau = {{1, 1, 1},
                              {1, 1, 1}};

    Tensor2D beta_hat_tau = {{1, 1, 1},
                             {1, 1, 1}};

    Parameters parameters(n_resolutions, n_factors, n_features,
                          Y,
                          log_p_pi,
                          log_p_eta,
                          alpha_t,
                          beta_t,
                          alpha_tau,
                          beta_tau,
                          mu_L,
                          sigma_squared_L,
                          log_r_pi,
                          mu_F,
                          sigma_squared_F,
                          log_r_eta,
                          alpha_hat_t,
                          beta_hat_t,
                          alpha_hat_tau,
                          beta_hat_tau);

    // Mock values for testing utilities
    const double gamma_t_i_l = 1;
    const double gamma_tau_i_l = 1;
    const double xi_L_ijk_l = 0.5;
    const double xi_F_i_j = 0.5;
    const double lambda_L_ijk_l = 0.5 * 2;
    const double lambda_F_i_j = 0.5 * 2;
    const double theta_t_i_l = Eigen::numext::digamma(1.0) - std::log(2 * M_PI);
    const double theta_tau_i_l = Eigen::numext::digamma(1.0) - std::log(2 * M_PI);
    const double u_F_i_l_d = 0.5 * 2;
    const double v_F_ijk_l_d = 0.5;
    const double w_F_ijk_l_d = 0.5 * 0.25;
    const double s_F_ijk_l_d = 0.5 - 0.5 * 0.25;
    const double s_bar_F_ijk_l = (0.5 - 0.5 * 0.25) * 3;
    const double u_bar_F_i_l = 0.5 * 2 * 3;
    const double u_L_abc_i_j = 0.5 * 2;
    const double v_L_abc_i_j = 0.5;
    const double w_L_abc_i_j = 0.5 * 0.25;
    const double s_L_abc_i_j = 0.5 - 0.5 * 0.25;
    const double s_bar_L_i_j = (0.5 - 0.5 * 0.25) * 16;
    const double u_bar_L_i_j = 0.5 * 2 * 16;

    // Mock values for testing updates
    const double update_sigma_squared_L_ijk_l = 1.0 / (1 + 0.5 * 2 * 3);
    const double update_mu_L_ijk_l = s_bar_F_ijk_l * update_sigma_squared_L_ijk_l;
    const double update_pi_ijk_l_log_relative_pmf_0 = std::log(0.5);
    const double update_pi_ijk_l_log_relative_pmf_1 = std::log(
        std::sqrt(2 * M_PI * update_sigma_squared_L_ijk_l) *
        std::exp(0.5 * (theta_t_i_l + std::pow(update_mu_L_ijk_l, 2) / update_sigma_squared_L_ijk_l)) *
        0.5);
    const double update_log_r_pi_ijk_l = std::log(
        std::exp(update_pi_ijk_l_log_relative_pmf_1) /
        (std::exp(update_pi_ijk_l_log_relative_pmf_0) + std::exp(update_pi_ijk_l_log_relative_pmf_1)));
    const double update_sigma_squared_F_i_j = 1.0 / (1 + 0.5 * 2 * 16);
    const double update_mu_F_i_j = s_bar_L_i_j * update_sigma_squared_F_i_j;
    const double update_eta_i_j_log_relative_pmf_0 = std::log(0.5);
    const double update_eta_i_j_log_relative_pmf_1 = std::log(
        std::sqrt(2 * M_PI * update_sigma_squared_F_i_j) *
        std::exp(0.5 * (-std::log(2 * M_PI) + std::pow(update_mu_F_i_j, 2) / update_sigma_squared_F_i_j)) *
        0.5);
    const double update_log_r_eta_i_j = std::log(
        std::exp(update_eta_i_j_log_relative_pmf_1) /
        (std::exp(update_eta_i_j_log_relative_pmf_0) + std::exp(update_eta_i_j_log_relative_pmf_1)));

    const double update_alpha_hat_tau_i_l_0 = 4.0 / 2 + 1;
    const double update_alpha_hat_tau_i_l_1 = 12.0 / 2 + 1;
    const double update_beta_hat_tau_i_l_0 = 1 + 0.5 * (4 - 2 * 4 * 2 * 0.25 + 4 * 2 * (1 - 0.25 * 0.25) + 4 * std::pow(2 * 0.25, 2));
    const double update_beta_hat_tau_i_l_1 = 1 + 0.5 * (12 - 2 * 12 * 2 * 0.25 + 12 * 2 * (1 - 0.25 * 0.25) + 12 * std::pow(2 * 0.25, 2));

    const double update_alpha_hat_t_i_l_0 = 0.5 * 4 / 2 + 1;
    const double update_alpha_hat_t_i_l_1 = 0.5 * 12 / 2 + 1;
    const double update_beta_hat_t_i_l_0 = 1 + 0.5 * 4;
    const double update_beta_hat_t_i_l_1 = 1 + 0.5 * 12;

    // For testing cavi_elbo
    const double E_log_likelihood_Y_ijk_l_given_pi_L_F_tau = 0.5 * (theta_tau_i_l - gamma_tau_i_l * (1.0 - 2 * 2 * 0.5 * 0.5 + 2 * std::pow(0.5, 4) + 2 * 0.5 * 2 * 0.5 * 2));
    const double E_log_likelihood_L_ijk_l_given_pi_t = 0.5 * (0.5 * theta_t_i_l - gamma_t_i_l * lambda_L_ijk_l);
    const double E_log_likelihood_F_i_j_given_eta = -0.5 * (0.5 * std::log(2 * M_PI) + lambda_F_i_j);
    const double E_log_likelihood_pi_ijk_l = 0.5 * std::log(0.5) + (1 - 0.5) * std::log(1 - 0.5);
    const double E_log_likelihood_eta_i_j = 0.5 * std::log(0.5) + (1 - 0.5) * std::log(1 - 0.5);
    const double E_log_likelihood_t_i_l = -gamma_t_i_l - std::lgamma(1);
    const double E_log_likelihood_tau_i_l = -gamma_tau_i_l - std::lgamma(1);
    const double E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l = ((0.5 / 2) * std::log(2 * M_PI * 1 + 1) - 0.5 * std::log(0.5) - (1 - 0.5) * std::log(1 - 0.5));
    const double E_negative_variational_log_likelihood_F_i_j_eta_i_j = ((0.5 / 2) * std::log(2 * M_PI * 1 + 1) - 0.5 * std::log(0.5) - (1 - 0.5) * std::log(1 - 0.5));
    const double E_negative_variational_log_likelihood_t_i_l = 1 + std::lgamma(1);
    const double E_negative_variational_log_likelihood_tau_i_l = 1 + std::lgamma(1);
    const double elbo = (16 * 3 * E_log_likelihood_Y_ijk_l_given_pi_L_F_tau + 16 * 2 * E_log_likelihood_L_ijk_l_given_pi_t + 2 * 3 * E_log_likelihood_F_i_j_given_eta + 16 * 2 * E_log_likelihood_pi_ijk_l + 2 * 3 * E_log_likelihood_eta_i_j + 2 * 2 * E_log_likelihood_t_i_l + 2 * 3 * E_log_likelihood_tau_i_l + 16 * 2 * E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l + 2 * 3 * E_negative_variational_log_likelihood_F_i_j_eta_i_j + 2 * 2 * E_negative_variational_log_likelihood_t_i_l + 2 * 3 * E_negative_variational_log_likelihood_tau_i_l);
}