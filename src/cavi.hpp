#ifndef CAVI_HPP_INCLUDED
#define CAVI_HPP_INCLUDED

#include "parameters.hpp"
#include "updates.hpp"
#include "elbo.hpp"
#include <limits>
#include "init.hpp"
#include <vector>
#include <cmath>
#include "tensor.hpp"
#include "init.hpp"

struct CaviResult
{
    Parameters parameters;
    std::vector<double> elbo_record;
    double elbo;
};

class CaviDimensions
{
public:
    // Constructor with initialization list
    CaviDimensions(int factors, int resolutions, int features, int spots);

    // Member variables
    const int n_factors;
    const int n_resolutions;
    const int n_features;
    const int n_spots;
    const int p_pi_shape;
    const int p_eta_shape;
    const std::array<int, 2> ab_t_shape;
    const std::array<int, 2> ab_tau_shape;
    const std::array<int, 2> F_shape;
    const Tensor4D L_skeleton;
    const Tensor4D Y_skeleton;

private:
    // Helper functions to initialize L_skeleton and Y_skeleton
    Tensor4D initLSkeleton() const;
    Tensor4D initYSkeleton() const;
};

CaviResult cavi(Parameters &parameters, int max_iterations, double relative_elbo_threshold);

#endif /*CAVI_HPP_INCLUDED*/