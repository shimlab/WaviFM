#include "parameters.hpp"

// Constructor to initialize all fields based on provided values
Parameters::Parameters(int n_resolutions_init, int n_factors_init, int n_features_init,
                       const Tensor4D &Y_init,
                       const Tensor1D &log_p_pi_init,
                       const Tensor1D &log_p_eta_init,
                       const Tensor2D &alpha_t_init,
                       const Tensor2D &beta_t_init,
                       const Tensor2D &alpha_tau_init,
                       const Tensor2D &beta_tau_init,
                       const Tensor4D &mu_L_init,
                       const Tensor4D &sigma_squared_L_init,
                       const Tensor4D &log_r_pi_init,
                       const Tensor2D &mu_F_init,
                       const Tensor2D &sigma_squared_F_init,
                       const Tensor2D &log_r_eta_init,
                       const Tensor2D &alpha_hat_t_init,
                       const Tensor2D &beta_hat_t_init,
                       const Tensor2D &alpha_hat_tau_init,
                       const Tensor2D &beta_hat_tau_init)
    : n_resolutions(n_resolutions_init), n_factors(n_factors_init), n_features(n_features_init),
      Y(Y_init), log_p_pi(log_p_pi_init), log_p_eta(log_p_eta_init),
      alpha_t(alpha_t_init), beta_t(beta_t_init),
      alpha_tau(alpha_tau_init), beta_tau(beta_tau_init),
      mu_L(mu_L_init), sigma_squared_L(sigma_squared_L_init), log_r_pi(log_r_pi_init),
      mu_F(mu_F_init), sigma_squared_F(sigma_squared_F_init), log_r_eta(log_r_eta_init),
      alpha_hat_t(alpha_hat_t_init), beta_hat_t(beta_hat_t_init),
      alpha_hat_tau(alpha_hat_tau_init), beta_hat_tau(beta_hat_tau_init)
{
    // Constructor body is empty as all fields are initialized in the initializer list
}

// Copy constructor for deep copying
Parameters::Parameters(const Parameters &other)
    : n_resolutions(other.n_resolutions),
      n_factors(other.n_factors),
      n_features(other.n_features),
      Y(other.Y),
      log_p_pi(other.log_p_pi),
      log_p_eta(other.log_p_eta),
      alpha_t(other.alpha_t),
      beta_t(other.beta_t),
      alpha_tau(other.alpha_tau),
      beta_tau(other.beta_tau),
      mu_L(other.mu_L),
      sigma_squared_L(other.sigma_squared_L),
      log_r_pi(other.log_r_pi),
      mu_F(other.mu_F),
      sigma_squared_F(other.sigma_squared_F),
      log_r_eta(other.log_r_eta),
      alpha_hat_t(other.alpha_hat_t),
      beta_hat_t(other.beta_hat_t),
      alpha_hat_tau(other.alpha_hat_tau),
      beta_hat_tau(other.beta_hat_tau) {}

// Copy assignment operator for deep copying
Parameters &Parameters::operator=(const Parameters &other)
{
    if (this == &other)
        return *this; // Handle self-assignment

    n_resolutions = other.n_resolutions;
    n_factors = other.n_factors;
    n_features = other.n_features;
    Y = other.Y;
    log_p_pi = other.log_p_pi;
    log_p_eta = other.log_p_eta;
    alpha_t = other.alpha_t;
    beta_t = other.beta_t;
    alpha_tau = other.alpha_tau;
    beta_tau = other.beta_tau;
    mu_L = other.mu_L;
    sigma_squared_L = other.sigma_squared_L;
    log_r_pi = other.log_r_pi;
    mu_F = other.mu_F;
    sigma_squared_F = other.sigma_squared_F;
    log_r_eta = other.log_r_eta;
    alpha_hat_t = other.alpha_hat_t;
    beta_hat_t = other.beta_hat_t;
    alpha_hat_tau = other.alpha_hat_tau;
    beta_hat_tau = other.beta_hat_tau;

    return *this;
}