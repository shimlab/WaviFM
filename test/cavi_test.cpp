#include <gtest/gtest.h>
#include "mocks.hpp"
#include "cavi.hpp"
#include "testing_utilities.hpp"

TEST(CaviTest, CaviDimensionsTest)
{
    CaviDimensions dim = CaviDimensions(2, 3, 3, 64);

    EXPECT_EQ(dim.n_factors, 2);
    EXPECT_EQ(dim.n_resolutions, 3);
    EXPECT_EQ(dim.n_features, 3);
    EXPECT_EQ(dim.n_spots, 64);
    EXPECT_EQ(dim.p_pi_shape, 3);
    EXPECT_EQ(dim.p_eta_shape, 2);
    EXPECT_EQ(dim.ab_t_shape[0], 3);
    EXPECT_EQ(dim.ab_t_shape[1], 2);
    EXPECT_EQ(dim.ab_tau_shape[0], 3);
    EXPECT_EQ(dim.ab_tau_shape[1], 3);
    EXPECT_EQ(dim.F_shape[0], 2);
    EXPECT_EQ(dim.F_shape[1], 3);

    // Check dimensions of L_skeleton
    EXPECT_EQ(dim.L_skeleton.size(), 2); // Should have n_factors elements
    for (int l = 0; l < 2; ++l)
    {
        EXPECT_EQ(dim.L_skeleton[l].size(), 3); // Should have n_resolutions elements

        int i = 0;
        EXPECT_EQ(dim.L_skeleton[l][i].size(), 1); // Approx level coef, should only have 1 set of coefficients
        for (int j = 0; j < 1; ++j)
        {
            EXPECT_EQ(dim.L_skeleton[l][i][j].size(), 4);
        }

        i = 1;
        EXPECT_EQ(dim.L_skeleton[l][i].size(), 3); // Detail level coef, should have 3 sets of coefficients
        for (int j = 0; j < 1; ++j)
        {
            EXPECT_EQ(dim.L_skeleton[l][i][j].size(), 4);
        }

        i = 2;
        EXPECT_EQ(dim.L_skeleton[l][i].size(), 3); // Detail level coef, should have 3 sets of coefficients
        for (int j = 0; j < 1; ++j)
        {
            EXPECT_EQ(dim.L_skeleton[l][i][j].size(), 16);
        }
    }
}

TEST(CaviTest, Cavi)
{
    // Obtain cavi fit values
    CaviResult res = cavi(mocks::parameters, 1000, 0.0001);

    // True cavi fit values
    double true_elbo = -83.41810231179173;
    std::vector<double> true_elbo_record = {-131.17688406881805, -83.4888455289592, -83.41810231179173};
    Parameters true_parameters(
        2, // n_resolutions
        2, // n_factors
        3, // n_features
        {{{{1, 1, 1, 1}}, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}},
         {{{1, 1, 1, 1}}, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}},
         {{{1, 1, 1, 1}}, {{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}}}}, // Y
        {-0.69314718, -0.69314718},                                     // log_p_pi
        {-0.69314718, -0.69314718},                                     // log_p_eta
        {{1, 1}, {1, 1}},                                               // alpha_t
        {{1, 1}, {1, 1}},                                               // beta_t
        {{1, 1, 1}, {1, 1, 1}},                                         // alpha_tau
        {{1, 1, 1}, {1, 1, 1}},                                         // beta_tau
        {{{{0.30304014, 0.30304014, 0.30304014, 0.30304014}},
          {{0.25575863, 0.25575863, 0.25575863, 0.25575863},
           {0.25575863, 0.25575863, 0.25575863, 0.25575863},
           {0.25575863, 0.25575863, 0.25575863, 0.25575863}}},
         {{{0.38185119, 0.38185119, 0.38185119, 0.38185119}},
          {{0.33361385, 0.33361385, 0.33361385, 0.33361385},
           {0.33361385, 0.33361385, 0.33361385, 0.33361385},
           {0.33361385, 0.33361385, 0.33361385, 0.33361385}}}}, // mu_L
        {{{{0.44794941, 0.44794941, 0.44794941, 0.44794941}},
          {{0.37533124, 0.37533124, 0.37533124, 0.37533124},
           {0.37533124, 0.37533124, 0.37533124, 0.37533124},
           {0.37533124, 0.37533124, 0.37533124, 0.37533124}}},
         {{{0.41003761, 0.41003761, 0.41003761, 0.41003761}},
          {{0.3545829, 0.3545829, 0.3545829, 0.3545829},
           {0.3545829, 0.3545829, 0.3545829, 0.3545829},
           {0.3545829, 0.3545829, 0.3545829, 0.3545829}}}}, // sigma_squared_L
        {{{{-0.86774981, -0.86774981, -0.86774981, -0.86774981}},
          {{-0.8046856, -0.8046856, -0.8046856, -0.8046856},
           {-0.8046856, -0.8046856, -0.8046856, -0.8046856},
           {-0.8046856, -0.8046856, -0.8046856, -0.8046856}}},
         {{{-0.8504464, -0.8504464, -0.8504464, -0.8504464}},
          {{-0.79034998, -0.79034998, -0.79034998, -0.79034998},
           {-0.79034998, -0.79034998, -0.79034998, -0.79034998},
           {-0.79034998, -0.79034998, -0.79034998, -0.79034998}}}},                         // log_r_pi
        {{0.42218012, 0.42218012, 0.42218012}, {0.54261435, 0.54261435, 0.54261435}},       // mu_F
        {{0.22871761, 0.22871761, 0.22871761}, {0.21771443, 0.21771443, 0.21771443}},       // sigma_squared_F
        {{-0.88220437, -0.88220437, -0.88220437}, {-0.73712397, -0.73712397, -0.73712397}}, // log_r_eta
        {{1.83979066, 1.85444835}, {3.68337108, 3.72211593}},                               // alpha_hat_t
        {{1.4533045, 1.47494336}, {2.18267894, 2.26818237}},                                // beta_hat_t
        {{3, 3, 3}, {7, 7, 7}},                                                             // alpha_hat_tau
        {{2.93814639, 2.93814639, 2.93814639}, {6.80799685, 6.80799685, 6.80799685}}        // beta_hat_tau
    );

    CaviResult true_res = {true_parameters, true_elbo_record, true_elbo};

    // Compare cavi output with true values
    const double precision = 0.001;
    EXPECT_EQ(true_res.parameters.n_resolutions, res.parameters.n_resolutions);
    EXPECT_EQ(true_res.parameters.n_factors, res.parameters.n_factors);
    EXPECT_EQ(true_res.parameters.n_features, res.parameters.n_features);
    compareTensor4D(true_res.parameters.Y, res.parameters.Y, precision);
    compareTensor1D(true_res.parameters.log_p_pi, res.parameters.log_p_pi, precision);
    compareTensor1D(true_res.parameters.log_p_eta, res.parameters.log_p_eta, precision);
    compareTensor2D(true_res.parameters.alpha_t, res.parameters.alpha_t, precision);
    compareTensor2D(true_res.parameters.beta_t, res.parameters.beta_t, precision);
    compareTensor2D(true_res.parameters.alpha_tau, res.parameters.alpha_tau, precision);
    compareTensor2D(true_res.parameters.beta_tau, res.parameters.beta_tau, precision);
    compareTensor4D(true_res.parameters.mu_L, res.parameters.mu_L, precision);
    compareTensor4D(true_res.parameters.sigma_squared_L, res.parameters.sigma_squared_L, precision);
    compareTensor4D(true_res.parameters.log_r_pi, res.parameters.log_r_pi, precision);
    compareTensor2D(true_res.parameters.mu_F, res.parameters.mu_F, precision);
    compareTensor2D(true_res.parameters.sigma_squared_F, res.parameters.sigma_squared_F, precision);
    compareTensor2D(true_res.parameters.log_r_eta, res.parameters.log_r_eta, precision);
    compareTensor2D(true_res.parameters.alpha_hat_t, res.parameters.alpha_hat_t, precision);
    compareTensor2D(true_res.parameters.beta_hat_t, res.parameters.beta_hat_t, precision);
    compareTensor2D(true_res.parameters.alpha_hat_tau, res.parameters.alpha_hat_tau, precision);
    compareTensor2D(true_res.parameters.beta_hat_tau, res.parameters.beta_hat_tau, precision);

    EXPECT_NEAR(true_res.elbo, res.elbo, precision);
    compareVectorDouble(true_res.elbo_record, res.elbo_record, precision);
}