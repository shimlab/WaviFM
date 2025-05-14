#include <gtest/gtest.h>
#include "updates.hpp"
#include "mocks.hpp"
#include <unsupported/Eigen/SpecialFunctions>

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredL)
{
    EXPECT_NEAR(compute_update_sigma_squared_L(1, 1, 1, 1, mocks::parameters), mocks::update_sigma_squared_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateMuL)
{
    EXPECT_NEAR(compute_update_mu_L(1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::parameters), mocks::update_mu_L_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdatePiIjkLRelativePmf)
{
    EXPECT_NEAR(compute_update_pi_ijk_l_log_relative_pmf(1, 1, 1, 1, 0, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_pi_ijk_l_log_relative_pmf_0, 0.001);
    EXPECT_NEAR(compute_update_pi_ijk_l_log_relative_pmf(1, 1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_pi_ijk_l_log_relative_pmf_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateRPi)
{
    EXPECT_NEAR(compute_update_log_r_pi(1, 1, 1, 1, mocks::update_sigma_squared_L_ijk_l, mocks::update_mu_L_ijk_l, mocks::parameters), mocks::update_log_r_pi_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateLPi)
{
    auto update_L_pi = compute_update_L_pi(1, 1, 1, 1, mocks::parameters);
    EXPECT_NEAR(update_L_pi.sigma_squared_L, mocks::update_sigma_squared_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L_pi.mu_L, mocks::update_mu_L_ijk_l, 0.001);
    EXPECT_NEAR(update_L_pi.log_r_pi, mocks::update_log_r_pi_ijk_l, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateSigmaSquaredF)
{
    EXPECT_NEAR(compute_update_sigma_squared_F(1, 1, mocks::parameters), mocks::update_sigma_squared_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateMuF)
{
    EXPECT_NEAR(compute_update_mu_F(1, 1, mocks::update_sigma_squared_F_i_j, mocks::parameters), mocks::update_mu_F_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateEtaIJRelativePmf)
{
    EXPECT_NEAR(compute_update_eta_i_j_log_relative_pmf(1, 1, 0, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_eta_i_j_log_relative_pmf_0, 0.001);
    EXPECT_NEAR(compute_update_eta_i_j_log_relative_pmf(1, 1, 1, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_eta_i_j_log_relative_pmf_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateREta)
{
    EXPECT_NEAR(compute_update_log_r_eta(1, 1, mocks::update_sigma_squared_F_i_j, mocks::update_mu_F_i_j, mocks::parameters), mocks::update_log_r_eta_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateFEta)
{
    auto update_F_eta = compute_update_F_eta(1, 1, mocks::parameters);
    EXPECT_NEAR(update_F_eta.sigma_squared_F, mocks::update_sigma_squared_F_i_j, 0.001);
    EXPECT_NEAR(update_F_eta.mu_F, mocks::update_mu_F_i_j, 0.001);
    EXPECT_NEAR(update_F_eta.log_r_eta, mocks::update_log_r_eta_i_j, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateAlphaHatTau)
{
    EXPECT_NEAR(compute_update_alpha_hat_tau(0, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_alpha_hat_tau(1, 1, mocks::parameters), mocks::update_alpha_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateBetaHatTau)
{
    EXPECT_NEAR(compute_update_beta_hat_tau(0, 1, mocks::parameters), mocks::update_beta_hat_tau_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_beta_hat_tau(1, 1, mocks::parameters), mocks::update_beta_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateTau)
{
    auto update_tau = compute_update_tau(1, 1, mocks::parameters);
    EXPECT_NEAR(update_tau.alpha_hat_tau, mocks::update_alpha_hat_tau_i_l_1, 0.001);
    EXPECT_NEAR(update_tau.beta_hat_tau, mocks::update_beta_hat_tau_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateAlphaHatT)
{
    EXPECT_NEAR(compute_update_alpha_hat_t(0, 1, mocks::parameters), mocks::update_alpha_hat_t_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_alpha_hat_t(1, 1, mocks::parameters), mocks::update_alpha_hat_t_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateBetaHatT)
{
    EXPECT_NEAR(compute_update_beta_hat_t(0, 1, mocks::parameters), mocks::update_beta_hat_t_i_l_0, 0.001);
    EXPECT_NEAR(compute_update_beta_hat_t(1, 1, mocks::parameters), mocks::update_beta_hat_t_i_l_1, 0.001);
}

TEST(CaviUpdatesTest, ComputeUpdateT)
{
    auto update_t = compute_update_t(1, 1, mocks::parameters);
    EXPECT_NEAR(update_t.alpha_hat_t, mocks::update_alpha_hat_t_i_l_1, 0.001);
    EXPECT_NEAR(update_t.beta_hat_t, mocks::update_beta_hat_t_i_l_1, 0.001);
}