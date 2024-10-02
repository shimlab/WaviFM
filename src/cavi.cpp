#include "cavi.hpp"

CaviDimensions::CaviDimensions(int factors, int resolutions, int features, int spots)
    : n_factors(factors),
      n_resolutions(resolutions),
      n_features(features),
      n_spots(spots),
      p_pi_shape(resolutions),
      p_eta_shape(factors),
      ab_t_shape({resolutions, factors}),
      ab_tau_shape({resolutions, features}),
      F_shape({factors, features}),
      L_skeleton(initLSkeleton()),
      Y_skeleton(initYSkeleton())
{
    // Check if n_resolutions is greater than or equal to 2 (approx level resolution and one detail level resolution)
    if (n_resolutions < 2)
    {
        throw std::invalid_argument("n_resolutions must be greater than or equal to 2");
    }

    // Check if n_spots is a power of 4
    int log4_spots = static_cast<int>(std::log2(n_spots) / 2);
    if (std::pow(4, log4_spots) != n_spots)
    {
        throw std::invalid_argument("n_spots must be a power of 4");
    }

    // Check if n_spots >= 4^(n_resolutions-1)
    if (n_spots < static_cast<int>(std::pow(4, n_resolutions - 1)))
    {
        throw std::invalid_argument("n_spots must be greater than or equal to 4^(n_resolutions-1)");
    }
}

Tensor4D CaviDimensions::initLSkeleton() const
{
    Tensor4D skeleton;
    for (size_t l = 0; l < n_factors; ++l)
    {
        Tensor3D factor_element;
        for (size_t i = 0; i < n_resolutions; ++i)
        {
            Tensor2D resolution_element;
            int n_wavelet_types;
            int n_values;
            if (i == 0)
            {
                // For approx level wavelet coefficients
                n_wavelet_types = 1;
                n_values = n_spots / std::pow(4, n_resolutions - 1);
            }
            else
            {
                // For detail level wavelet coefficients
                n_wavelet_types = 3;
                n_values = n_spots / std::pow(4, n_resolutions - i);
            }
            for (size_t j = 0; j < n_wavelet_types; ++j)
            {
                resolution_element.push_back(Tensor1D(n_values, 0.0));
            }
            factor_element.push_back(resolution_element);
        }
        skeleton.push_back(factor_element);
    }
    return skeleton;
}

Tensor4D CaviDimensions::initYSkeleton() const
{
    Tensor4D skeleton;
    for (size_t l = 0; l < n_features; ++l)
    {
        Tensor3D feature_element;
        for (size_t i = 0; i < n_resolutions; ++i)
        {
            Tensor2D resolution_element;
            int n_wavelet_types;
            if (i == 0)
            {
                n_wavelet_types = 1; // For approx level wavelet coefficients
            }
            else
            {
                n_wavelet_types = 3; // For detail level wavelet coefficients
            }
            for (size_t j = 0; j < n_wavelet_types; ++j)
            {
                int n_values;
                if (i == 0)
                {
                    n_values = n_spots / std::pow(4, n_resolutions - 1); // For approx level wavelet coefficients
                }
                else
                {
                    n_values = n_spots / std::pow(4, n_resolutions - i); // For detail level wavelet coefficients
                }
                resolution_element.push_back(Tensor1D(n_values, 0.0));
            }
            feature_element.push_back(resolution_element);
        }
        skeleton.push_back(feature_element);
    }
    return skeleton;
}

CaviResult cavi(Parameters &parameters, int max_iterations, double relative_elbo_threshold)
{
    Tensor4D Y = parameters.Y;
    int num_iterations_completed = 0;
    double elbo = compute_elbo(parameters);
    std::vector<double> elbo_record;
    elbo_record.reserve(max_iterations + 1); // Reserve space to prevent reallocations, in turn speed up code. +1 to also store the initial elbo value (aka "iteration 0")
    elbo_record.push_back(elbo);
    double prev_elbo = elbo;

    int n_factors = parameters.n_factors;
    int n_features = parameters.n_features;
    int n_resolutions = parameters.n_resolutions;

    while (num_iterations_completed < max_iterations)
    {
        // Run one iteration of CAVI updates
        Parameters new_parameters = parameters; // Copy construction

        // For L_ijk_l, pi_ijk_l related updates
        for (int l = 0; l < n_factors; ++l)
        {
            for (int i = 0; i < n_resolutions; ++i)
            {
                for (std::size_t j = 0; j < Y[l][i].size(); ++j)
                {
                    for (std::size_t k = 0; k < Y[l][i][j].size(); ++k)
                    {
                        UpdateLPIResult L_pi_ijk_l_update = compute_update_L_pi(i, j, k, l, new_parameters);
                        new_parameters.sigma_squared_L[l][i][j][k] = L_pi_ijk_l_update.sigma_squared_L;
                        new_parameters.mu_L[l][i][j][k] = L_pi_ijk_l_update.mu_L;
                        new_parameters.log_r_pi[l][i][j][k] = L_pi_ijk_l_update.log_r_pi;
                    }
                }
            }
        }

        // For F_i_j, eta_i_j related updates
        for (int i = 0; i < n_factors; ++i)
        {
            for (int j = 0; j < n_features; ++j)
            {
                UpdateFEtaResult F_eta_ij_update = compute_update_F_eta(i, j, new_parameters);
                new_parameters.sigma_squared_F[i][j] = F_eta_ij_update.sigma_squared_F;
                new_parameters.mu_F[i][j] = F_eta_ij_update.mu_F;
                new_parameters.log_r_eta[i][j] = F_eta_ij_update.log_r_eta;
            }
        }

        // For tau_i_l related updates
        for (int i = 0; i < n_resolutions; ++i)
        {
            for (int l = 0; l < n_features; ++l)
            {
                UpdateTauResult tau_i_l_update = compute_update_tau(i, l, new_parameters);
                new_parameters.alpha_hat_tau[i][l] = tau_i_l_update.alpha_hat_tau;
                new_parameters.beta_hat_tau[i][l] = tau_i_l_update.beta_hat_tau;
            }
        }

        // For t_i_l related updates
        for (int i = 0; i < n_resolutions; ++i)
        {
            for (int l = 0; l < n_factors; ++l)
            {
                UpdateTResult t_i_l_update = compute_update_t(i, l, new_parameters);
                new_parameters.alpha_hat_t[i][l] = t_i_l_update.alpha_hat_t;
                new_parameters.beta_hat_t[i][l] = t_i_l_update.beta_hat_t;
            }
        }

        // Discard current iteration and terminate if elbo dropped
        elbo = compute_elbo(new_parameters);
        if (elbo < prev_elbo)
        {
            break;
        }

        // Accept current iteration and increment iteration counter
        ++num_iterations_completed;
        elbo_record.push_back(elbo);
        parameters = new_parameters; // Copy assignment

        // Terminate if ELBO converged
        if (std::abs(prev_elbo) > 0 && std::isfinite(prev_elbo)) // >0 is there to avoid division by 0 when computing relative_diff_elbo
        {
            double relative_diff_elbo = std::abs((elbo - prev_elbo) / prev_elbo);
            if (relative_diff_elbo < relative_elbo_threshold)
            {
                break;
            }
        }

        // Record current ELBO for comparison in the next iteration
        prev_elbo = elbo;
    }

    return CaviResult{
        parameters,
        elbo_record,
        elbo_record.back()};
}