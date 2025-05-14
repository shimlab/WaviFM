#include <gtest/gtest.h>
#include "elbo.hpp"
#include "mocks.hpp"
#include <cmath>

TEST(CaviElboTest, ComputeELogLikelihoodLijkLGivenPiT)
{
    EXPECT_NEAR(compute_E_log_likelihood_Y_ijk_l_given_pi_L_F_tau(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_Y_ijk_l_given_pi_L_F_tau, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodLijklGivenPiT)
{
    EXPECT_NEAR(compute_E_log_likelihood_L_ijk_l_given_pi_t(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_L_ijk_l_given_pi_t, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodFijGivenEta)
{
    EXPECT_NEAR(compute_E_log_likelihood_F_i_j_given_eta(1, 1, mocks::parameters), mocks::E_log_likelihood_F_i_j_given_eta, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodPiijkl)
{
    EXPECT_NEAR(compute_E_log_likelihood_pi_ijk_l(1, 1, 1, 1, mocks::parameters), mocks::E_log_likelihood_pi_ijk_l, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodEtaij)
{
    EXPECT_NEAR(compute_E_log_likelihood_eta_i_j(1, 1, mocks::parameters), mocks::E_log_likelihood_eta_i_j, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodTil)
{
    EXPECT_NEAR(compute_E_log_likelihood_t_i_l(1, 1, mocks::parameters), mocks::E_log_likelihood_t_i_l, 0.001);
}

TEST(CaviElboTest, ComputeELogLikelihoodTauil)
{
    EXPECT_NEAR(compute_E_log_likelihood_tau_i_l(1, 1, mocks::parameters), mocks::E_log_likelihood_tau_i_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodLijklPiijkl)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l(1, 1, 1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_L_ijk_l_pi_ijk_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodFijEtaij)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_F_i_j_eta_i_j(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_F_i_j_eta_i_j, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodTil)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_t_i_l(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_t_i_l, 0.001);
}

TEST(CaviElboTest, ComputeENegativeVariationalLogLikelihoodTauil)
{
    EXPECT_NEAR(compute_E_negative_variational_log_likelihood_tau_i_l(1, 1, mocks::parameters), mocks::E_negative_variational_log_likelihood_tau_i_l, 0.001);
}

TEST(CaviElboTest, ComputeElbo)
{
    EXPECT_NEAR(compute_elbo(mocks::parameters), mocks::elbo, 0.001);
}