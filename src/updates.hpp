#ifndef UPDATES_HPP_INCLUDED
#define UPDATES_HPP_INCLUDED

#include <cmath>
#include "utilities.hpp"
#include "parameters.hpp"

// Global constants
extern const double RELATIVE_PMF_INCREMENT;
extern const double LOG_RELATIVE_PMF_INCREMENT;

// Parameter update grouping structs
struct UpdateLPIResult
{
    double sigma_squared_L;
    double mu_L;
    double log_r_pi;
};

struct UpdateFEtaResult
{
    double sigma_squared_F;
    double mu_F;
    double log_r_eta;
};

struct UpdateTauResult
{
    double alpha_hat_tau;
    double beta_hat_tau;
};

struct UpdateTResult
{
    double alpha_hat_t;
    double beta_hat_t;
};

// For L_ijk_l, pi_ijk_l related updates
double compute_update_sigma_squared_L(int i, int j, int k, int l, const Parameters &parameters);
double compute_update_mu_L(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, const Parameters &parameters);
double compute_update_pi_ijk_l_log_relative_pmf(int i, int j, int k, int l, int pi, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters);
double compute_update_log_r_pi(int i, int j, int k, int l, double update_sigma_squared_L_ijk_l, double update_mu_L_ijk_l, const Parameters &parameters);
UpdateLPIResult compute_update_L_pi(int i, int j, int k, int l, const Parameters &parameters);

// For F_i_j, eta_i_j related updates
double compute_update_sigma_squared_F(int i, int j, const Parameters &parameters);
double compute_update_mu_F(int i, int j, double update_sigma_squared_F_i_j, const Parameters &parameters);
double compute_update_eta_i_j_log_relative_pmf(int i, int j, int eta, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters);
double compute_update_log_r_eta(int i, int j, double update_sigma_squared_F_i_j, double update_mu_F_i_j, const Parameters &parameters);
UpdateFEtaResult compute_update_F_eta(int i, int j, const Parameters &parameters);

// For tau_i_l related updates
double compute_update_alpha_hat_tau(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_tau(int i, int l, const Parameters &parameters);
UpdateTauResult compute_update_tau(int i, int l, const Parameters &parameters);
double compute_update_alpha_hat_t(int i, int l, const Parameters &parameters);
double compute_update_beta_hat_t(int i, int l, const Parameters &parameters);
UpdateTResult compute_update_t(int i, int l, const Parameters &parameters);

#endif /*UPDATES_HPP_INCLUDED*/