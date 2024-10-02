# This file contains code for running CAVI

# Code to enable this notebook to import from libraries
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from .cavi_updates import *
from .cavi_elbo import *
from .cavi_init import *
from .cavi_utilities import *
import build.WaviFM as WaviFM
import time
import copy


# Function to run the inference (pretty bad cos not very procedural as rely on a lot of global params, ceebs tbh just testing here anyways)
def cavi(parameters, max_iterations, relative_elbo_threshold, print_progress=True):
    Y = parameters["Y"]
    num_iterations_completed = 0
    prev_elbo = compute_elbo(parameters)
    elbo_record = [prev_elbo]

    n_factors = parameters["n_factors"]
    n_features = parameters["n_features"]
    n_resolutions = parameters["n_resolutions"]

    if print_progress:
        print(f"Initial ELBO: {prev_elbo}")

    while num_iterations_completed < max_iterations:
        if print_progress:
            print(
                "Start Iteration " + str(num_iterations_completed + 1), end=""
            )  # progress tracking print out

        # Run one iteration of CAVI updates

        new_parameters = copy.deepcopy(
            parameters
        )  # This new parameter is needed as opposed to in place update in case decide to discard say final update

        ## For L_ijk_l, pi_ijk_l related updates
        for l in range(n_factors):
            for i in range(n_resolutions):
                for j in range(len(Y[l][i])):
                    for k in range(len(Y[l][i][j])):
                        L_pi_ijk_l_update = compute_update_L_pi(
                            i, j, k, l, new_parameters
                        )
                        new_parameters["sigma_squared_L"][l][i][j][k] = (
                            L_pi_ijk_l_update["sigma_squared_L"]
                        )
                        new_parameters["mu_L"][l][i][j][k] = L_pi_ijk_l_update["mu_L"]
                        new_parameters["log_r_pi"][l][i][j][k] = L_pi_ijk_l_update[
                            "log_r_pi"
                        ]

        ## For F_i_j, eta_i_j related updates
        for i in range(n_factors):
            for j in range(n_features):
                F_eta_ij_update = compute_update_F_eta(i, j, new_parameters)
                new_parameters["sigma_squared_F"][i][j] = F_eta_ij_update[
                    "sigma_squared_F"
                ]
                new_parameters["mu_F"][i][j] = F_eta_ij_update["mu_F"]
                new_parameters["log_r_eta"][i][j] = F_eta_ij_update["log_r_eta"]

        ## For tau_i_l related updates
        for i in range(n_resolutions):
            for l in range(n_features):
                tau_i_l_update = compute_update_tau(i, l, new_parameters)
                new_parameters["alpha_hat_tau"][i][l] = tau_i_l_update["alpha_hat_tau"]
                new_parameters["beta_hat_tau"][i][l] = tau_i_l_update["beta_hat_tau"]

        ## For t_i_l related updates
        for i in range(n_resolutions):
            for l in range(n_factors):
                t_i_l_update = compute_update_t(i, l, new_parameters)
                new_parameters["alpha_hat_t"][i][l] = t_i_l_update["alpha_hat_t"]
                new_parameters["beta_hat_t"][i][l] = t_i_l_update["beta_hat_t"]

        # loop4_time = time.time() - start_time - loop1_time - loop2_time - loop3_time# Code for testing runtime

        # print(f"Loop 1 time: {loop1_time}, Loop 2 time: {loop2_time}, Loop 3 time: {loop3_time}, Loop 4 time: {loop4_time}")# Code for testing runtime

        # Discard current iteration and terminate if elbo dropped
        elbo = compute_elbo(new_parameters)
        if (
            elbo < prev_elbo
        ):  # Guard to avoid the strange situation where elbo decreases (likely due to numerical error)
            if print_progress:
                print(f" ELBO: {elbo} --- Discarded due to ELBO drop")
            break

        # Accept current iteration and increment iteration counter
        num_iterations_completed += 1
        elbo_record.append(elbo)
        parameters = new_parameters

        # Terminate if elbo converged
        if print_progress:
            print(f" ELBO: {elbo}")
        if np.abs(prev_elbo) > 0 and np.isfinite(prev_elbo):
            diff_elbo = np.abs(elbo - prev_elbo)
            relative_diff_elbo = np.abs(diff_elbo / prev_elbo)
            if relative_diff_elbo < relative_elbo_threshold:
                break

        # Record current elbo for comparison in next iteration
        prev_elbo = elbo

    return {
        "parameters": parameters,
        "elbo_record": elbo_record,
        "elbo": elbo_record[-1],
    }


# Function to run CAVI for multiple initialisations and choose result of the one with the best elbo
def cavi_multi_init(
    Y,
    dimensions,
    max_iterations,
    relative_elbo_threshold,
    n_init=5,
    print_progress=True,
    print_each_initialisation_progress=False,
):

    # Run CAVI for specified number of initialisations
    results_list = []
    for i in range(n_init):
        parameters = init_parameters(Y, dimensions)
        results = cavi(
            parameters,
            max_iterations,
            relative_elbo_threshold,
            print_progress=print_each_initialisation_progress,
        )
        results_list.append(results)
        elbo = results["elbo"]
        if print_progress:
            print(f"Initialisation {i+1}: ELBO = {elbo}")

    # Find and return the results from the CAVI run with the maximal ELBO
    results_with_max_elbo = max(results_list, key=lambda results: results["elbo"])
    return results_with_max_elbo


# Function to run CAVI for multiple initialisations and choose result of the one with the best elbo
def cavi_multi_init_cpp(
    Y,
    dimensions,
    max_iterations,
    relative_elbo_threshold,
    n_init=5,
    print_progress=True,
    priors=None,
):

    # Run CAVI for specified number of initialisations (uses the cpp implementation)
    results_list = []
    for i in range(n_init):
        parameters_cpp = init_parameters_cpp(Y, dimensions, priors)
        start = time.time()
        results_cpp = WaviFM.cavi(
            parameters_cpp, max_iterations, relative_elbo_threshold
        )
        end = time.time()
        cpp_time = end - start
        results = {
            "parameters": AttributeIndexer(results_cpp.parameters),
            "elbo_record": results_cpp.elbo_record,
            "elbo": results_cpp.elbo,
        }
        results_list.append(results)
        elbo = results["elbo"]
        if print_progress:
            n_iterations = (
                len(results["elbo_record"]) - 1
            )  # -1 since elbo record includes initial elbo before any iterations are done
            print(
                f"Initialisation {i+1}:\n\tELBO = {elbo}\n\t#Iterations = {n_iterations}\n\tTime taken (s) = {cpp_time}"
            )

    # Find and return the results from the CAVI run with the maximal ELBO
    index, results_with_max_elbo = max(
        enumerate(results_list), key=lambda x: x[1]["elbo"]
    )
    if print_progress:
        print(f"Initialisation {index+1} has maximal ELBO and is returned")
    return results_with_max_elbo