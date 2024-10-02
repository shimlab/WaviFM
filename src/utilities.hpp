#ifndef UTILITIES_HPP_INCLUDED
#define UTILITIES_HPP_INCLUDED

#include <cmath>
#include "parameters.hpp"
#include <unsupported/Eigen/SpecialFunctions>

double sum_log(double log_a, double log_b);
double gamma_t(int i, int l, const Parameters &parameters);
double gamma_tau(int i, int l, const Parameters &parameters);
double xi_L(int i, int j, int k, int l, const Parameters &parameters);
double xi_F(int i, int j, const Parameters &parameters);
double lambda_L(int i, int j, int k, int l, const Parameters &parameters);
double lambda_F(int i, int j, const Parameters &parameters);
double theta_t(int i, int l, const Parameters &parameters);
double theta_tau(int i, int l, const Parameters &parameters);
double u_F(int i, int l, int d, const Parameters &parameters);
double v_F(int i, int j, int k, int l, int d, const Parameters &parameters);
double w_F(int i, int j, int k, int l, int d, const Parameters &parameters);
double s_F(int i, int j, int k, int l, int d, const Parameters &parameters);
double s_bar_F(int i, int j, int k, int l, const Parameters &parameters);
double u_bar_F(int i, int l, const Parameters &parameters);
double u_L(int a, int b, int c, int i, int j, const Parameters &parameters);
double v_L(int a, int b, int c, int i, int j, const Parameters &parameters);
double w_L(int a, int b, int c, int i, int j, const Parameters &parameters);
double s_L(int a, int b, int c, int i, int j, const Parameters &parameters);
double s_bar_L(int i, int j, const Parameters &parameters);
double u_bar_L(int i, int j, const Parameters &parameters);

#endif /*UTILITIES_HPP_INCLUDED*/