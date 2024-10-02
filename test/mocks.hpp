#ifndef MOCKS_HPP_INCLUDED
#define MOCKS_HPP_INCLUDED

#include "tensor.hpp"
#include "parameters.hpp"
#include <cmath>
#include <unsupported/Eigen/SpecialFunctions>

namespace mocks
{
    // Mock constants
    extern double log_half;

    // Mock dimension parameters
    extern int n_factors;
    extern int n_resolutions;
    extern int n_features;

    // Mock tensor parameters
    extern Tensor4D Y;
    extern Tensor1D log_p_pi;
    extern Tensor1D log_p_eta;
    extern Tensor2D alpha_t;
    extern Tensor2D beta_t;
    extern Tensor2D alpha_tau;
    extern Tensor2D beta_tau;
    extern Tensor4D mu_L;
    extern Tensor4D sigma_squared_L;
    extern Tensor4D log_r_pi;
    extern Tensor2D mu_F;
    extern Tensor2D sigma_squared_F;
    extern Tensor2D log_r_eta;
    extern Tensor2D alpha_hat_t;
    extern Tensor2D beta_hat_t;
    extern Tensor2D alpha_hat_tau;
    extern Tensor2D beta_hat_tau;

    // Mock Parameters instance
    extern Parameters parameters;

    // Mock return values for functions in utilities
    extern const double gamma_t_i_l;
    extern const double gamma_tau_i_l;
    extern const double xi_L_ijk_l;
    extern const double xi_F_i_j;
    extern const double lambda_L_ijk_l;
    extern const double lambda_F_i_j;
    extern const double theta_t_i_l;
    extern const double theta_tau_i_l;
    extern const double u_F_i_l_d;
    extern const double v_F_ijk_l_d;
    extern const double w_F_ijk_l_d;
    extern const double s_F_ijk_l_d;
    extern const double s_bar_F_ijk_l;
    extern const double u_bar_F_i_l;
    extern const double u_L_abc_i_j;
    extern const double v_L_abc_i_j;
    extern const double w_L_abc_i_j;
    extern const double s_L_abc_i_j;
    extern const double s_bar_L_i_j;
    extern const double u_bar_L_i_j;

    // Mock return values for functions in updates
    extern const double update_sigma_squared_L_ijk_l;
    extern const double update_mu_L_ijk_l;
    extern const double update_pi_ijk_l_log_relative_pmf_0;
    extern const double update_pi_ijk_l_log_relative_pmf_1;
    extern const double update_log_r_pi_ijk_l;
    extern const double update_sigma_squared_F_i_j;
    extern const double update_mu_F_i_j;
    extern const double update_eta_i_j_log_relative_pmf_0;
    extern const double update_eta_i_j_log_relative_pmf_1;
    extern const double update_log_r_eta_i_j;

    extern const double update_alpha_hat_tau_i_l_0;
    extern const double update_alpha_hat_tau_i_l_1;
    extern const double update_beta_hat_tau_i_l_0;
    extern const double update_beta_hat_tau_i_l_1;

    extern const double update_alpha_hat_t_i_l_0;
    extern const double update_alpha_hat_t_i_l_1;
    extern const double update_beta_hat_t_i_l_0;
    extern const double update_beta_hat_t_i_l_1;

    // For testing cavi_elbo
    extern const double E_log_likelihood_Y_ijk_l_given_pi_L_F_tau;
    extern const double E_log_likelihood_L_ijk_l_given_pi_t;
    extern const double E_log_likelihood_F_i_j_given_eta;
    extern const double E_log_likelihood_pi_ijk_l;
    extern const double E_log_likelihood_eta_i_j;
    extern const double E_log_likelihood_t_i_l;
    extern const double E_log_likelihood_tau_i_l;
    extern const double E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l;
    extern const double E_negative_variational_log_likelihood_F_i_j_eta_i_j;
    extern const double E_negative_variational_log_likelihood_t_i_l;
    extern const double E_negative_variational_log_likelihood_tau_i_l;
    extern const double elbo;
}

#endif /*MOCKS_HPP_INCLUDED*/