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