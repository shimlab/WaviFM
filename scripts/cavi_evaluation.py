# This file contain utilities to help evaluate cavi results

import numpy as np
import copy
from .cavi_updates import *
from .cavi_utilities import *

# Functions to compute estimations based on variational distribution
# {parameters} argument is the parameters output from cavi(.) call


def variational_approx_posterior_mean_L(parameters):
    n_factors = parameters["n_factors"]
    mean_values = copy.deepcopy(parameters["mu_L"])
    for l in range(n_factors):
        for i in range(len(parameters["mu_L"][l])):
            for j in range(len(parameters["mu_L"][l][i])):
                for k in range(len(parameters["mu_L"][l][i][j])):
                    mean_values[l][i][j][k] = xi_L(i, j, k, l, parameters)
    return mean_values


def variational_approx_posterior_mean_F(parameters):
    n_factors = parameters["n_factors"]
    n_features = parameters["n_features"]
    mean_values = copy.deepcopy(parameters["mu_F"])
    for i in range(n_factors):
        for j in range(n_features):
            mean_values[i][j] = xi_F(i, j, parameters)
    return np.array(mean_values)


def variational_approx_posterior_mean_pi(parameters):
    n_factors = parameters["n_factors"]
    mean_values = copy.deepcopy(parameters["log_r_pi"])
    for l in range(n_factors):
        for i in range(len(parameters["mu_L"][l])):
            for j in range(len(parameters["mu_L"][l][i])):
                for k in range(len(parameters["mu_L"][l][i][j])):
                    mean_values[l][i][j][k] = np.exp(parameters["log_r_pi"][l][i][j][k])
    return mean_values


def variational_approx_posterior_mean_eta(parameters):
    n_factors = parameters["n_factors"]
    n_features = parameters["n_features"]
    mean_values = copy.deepcopy(parameters["log_r_eta"])
    for i in range(n_factors):
        for j in range(n_features):
            mean_values[i][j] = np.exp(parameters["log_r_eta"][i][j])
    return np.array(mean_values)


# Functions to compute metrics to compare ground truths (assuming available) with inferred quantities

## Functions to compute RRMSE (similar to those defined in Foo & Shim 2021)


def nested_sum(estimate, truth, aggregator):
    # Computes aggregate of the two nested list-like structures {estimate}, {truth} using the provided aggregator function that allows for two arguments
    # Assumes estimate and truth are of the same type and shape (as nested list-like structures)

    aggregate = 0

    if isinstance(estimate, np.ndarray):
        estimate = estimate.tolist()

    if isinstance(truth, np.ndarray):
        truth = truth.tolist()

    if isinstance(estimate, (list, tuple)) and isinstance(truth, (list, tuple)):
        assert len(estimate) == len(truth)

        for est_item, tru_item in zip(estimate, truth):
            aggregate += nested_sum(est_item, tru_item, aggregator)
    else:
        aggregate += aggregator(estimate, truth)
    return aggregate


def nested_sum_squared_sum(estimate, truth):
    return nested_sum(estimate, truth, lambda estimate, truth: np.square(truth))


def nested_sum_squared_residuals(estimate, truth):
    return nested_sum(
        estimate, truth, lambda estimate, truth: np.square(estimate - truth)
    )


def rrmse(estimate, truth):
    # Computes a relative root mean squared error of the given nested list {estimate} like structure with the ground truth {truth} (which should have the same dimensions)
    # The computation uses an algorithm similar to those of RRMSE in Foo & Shim 2021
    # To accommodate diffent nested structures, a recursive approach to look into nested levels is used
    # The "relative" part of this RRMSE is the dividing by squared sum, which ensures a kind of scale independence of the metric so that it is comparable more or less

    sum_squared_residuals = nested_sum_squared_residuals(estimate, truth)
    sum_squared_sum = nested_sum_squared_sum(estimate, truth)
    return np.sqrt(sum_squared_residuals / sum_squared_sum)
