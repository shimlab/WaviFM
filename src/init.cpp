#include "init.hpp"

// #include "utilities.hpp"
// #include <xtensor/xrandom.hpp>
// #include <xtensor/xmath.hpp>
// #include <xtensor/xarray.hpp>

// Parameters init_parameters(const Tensor4D Y, const std::map<std::string, xt::xarray<double>>& dimensions) {
//     int n_resolutions = static_cast<int>(dimensions.n_resolutions);
//     int n_factors = static_cast<int>(dimensions.n_factors);
//     int n_features = static_cast<int>(dimensions.n_features);
//     auto p_pi_shape = dimensions.p_pi_shape;
//     auto p_eta_shape = dimensions.p_eta_shape;
//     auto F_shape = dimensions.F_shape;
//     auto ab_t_shape = dimensions.ab_t_shape;
//     auto ab_tau_shape = dimensions.ab_tau_shape;

//     // Initialise algorithm
//     auto mu_L = L_shaped_rand(0, 1, dimensions);
//     auto sigma_squared_L = L_shaped_rand(0, 1, dimensions);
//     auto log_r_pi = L_shaped_log_rand(0.01, 0.99, dimensions);
//     auto mu_F = xt::random::rand<double>(F_shape);
//     auto sigma_squared_F = xt::random::rand<double>(F_shape) + 0.01;
//     auto r_eta = xt::random::rand<double>(F_shape) * 0.98 + 0.01;
//     auto log_r_eta = xt::log(r_eta);
//     auto alpha_hat_t = xt::random::rand<double>(ab_t_shape) * 10 + 0.01;
//     auto beta_hat_t = xt::random::rand<double>(ab_t_shape) * 10 + 0.01;
//     auto alpha_hat_tau = xt::random::rand<double>(ab_tau_shape) * 10 + 0.01;
//     auto beta_hat_tau = xt::random::rand<double>(ab_tau_shape) * 10 + 0.01;

//     Parameters parameters{
//         {"n_resolutions", xt::xarray<double>({static_cast<double>(n_resolutions)})},
//         {"n_factors", xt::xarray<double>({static_cast<double>(n_factors)})},
//         {"n_features", xt::xarray<double>({static_cast<double>(n_features)})},
//         {"Y", Y}, // This and below are model hyperparameters
//         {"log_p_pi", xt::log(xt::full<double>(p_pi_shape, 0.3))},
//         {"log_p_eta", xt::log(xt::full<double>(p_eta_shape, 0.3))},
//         {"alpha_t", xt::full<double>(ab_t_shape, 1)},
//         {"beta_t", xt::full<double>(ab_t_shape, 1)},
//         {"alpha_tau", xt::full<double>(ab_tau_shape, 1)},
//         {"beta_tau", xt::full<double>(ab_tau_shape, 1)},
//         {"mu_L", mu_L}, // This and below are variational hyperparameters
//         {"sigma_squared_L", sigma_squared_L},
//         {"log_r_pi", log_r_pi},
//         {"mu_F", mu_F},
//         {"sigma_squared_F", sigma_squared_F},
//         {"log_r_eta", log_r_eta},
//         {"alpha_hat_t", alpha_hat_t},
//         {"beta_hat_t", beta_hat_t},
//         {"alpha_hat_tau", alpha_hat_tau},
//         {"beta_hat_tau", beta_hat_tau}
//     };

//     return parameters;
// }
