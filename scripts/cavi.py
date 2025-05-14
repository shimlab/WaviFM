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
from multiprocessing import Pool

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

def attribute_indexer_to_dict(indexer):
    """
    Recursively converts an AttributeIndexer object or a Pybind11-bound object 
    into a pure Python dictionary.

    This function is useful when the underlying object does not expose a __dict__ 
    attribute (e.g., C++ objects bound via Pybind11), but its public attributes 
    can still be accessed using dir() and getattr().

    Parameters
    ----------
    indexer : AttributeIndexer or object
        An instance of AttributeIndexer or a Pybind11-bound C++ object. If an 
        AttributeIndexer is passed, the function will extract from its internal `_obj`.

    Returns
    -------
    dict
        A dictionary representation of the object's accessible public attributes.
        If any attributes are nested AttributeIndexer instances, lists, or dicts, 
        they will be recursively converted as well.

    Notes
    -----
    - Private and special attributes (those starting with `_`) are ignored.
    - If an attribute is not readable (raises AttributeError), it is silently skipped.
    - This function supports nested structures of AttributeIndexers, lists, and dicts.
    """
    def unwrap(obj):
        if isinstance(obj, AttributeIndexer):
            return attribute_indexer_to_dict(obj)
        elif isinstance(obj, list):
            return [unwrap(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: unwrap(v) for k, v in obj.items()}
        else:
            return obj

    obj = indexer._obj if isinstance(indexer, AttributeIndexer) else indexer
    attr_names = [attr for attr in dir(obj) if not attr.startswith("_")]

    result = {}
    for attr in attr_names:
        try:
            value = getattr(obj, attr)
            result[attr] = unwrap(value)
        except AttributeError:
            continue  # skip if not accessible
    return result

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

# Functions to enable running CAVI for multiple initialisations in parallel to utilise multiprocessing, and choose result of the one with the best elbo
# Carefully designed to ensure iterations acknowledge random seed (either set explicitly or implicitly prior to running function) in the same manner as cavi_multi_init_cpp
def _run_cavi_from_python_params(parameters, max_iterations, relative_elbo_threshold):
    parameters_cpp = build_parameters_cpp(parameters)
    start = time.time()
    results_cpp = WaviFM.cavi(parameters_cpp, max_iterations, relative_elbo_threshold)
    end = time.time()
    cpp_time = end - start

    return {
        "parameters": attribute_indexer_to_dict(AttributeIndexer(results_cpp.parameters)), # Note this return differs from return of cavi_multi_init_cpp as here return a dictionary, albeit for analysis purpose very similar
        "elbo_record": results_cpp.elbo_record,
        "elbo": results_cpp.elbo,
        "cpp_time": cpp_time,
    }

def cavi_multi_init_cpp_parallel(
    Y,
    dimensions,
    max_iterations,
    relative_elbo_threshold,
    n_init=5,
    print_progress=True,
    priors=None,
):
    # Initialise all random parameter sets in main process to keep numpy seed behaviour consistent
    parameters_list = [init_parameters(Y, dimensions, priors) for _ in range(n_init)]

    # Run CAVI in parallel
    args_list = [(params, max_iterations, relative_elbo_threshold) for params in parameters_list]
    with Pool() as pool:
        results_list = pool.starmap(_run_cavi_from_python_params, args_list)

    # Print results
    if print_progress:
        for i, result in enumerate(results_list):
            n_iter = len(result["elbo_record"]) - 1
            print(
                f"Initialisation {i+1}:\n\tELBO = {result['elbo']}\n\t#Iterations = {n_iter}\n\tTime taken (s) = {result['cpp_time']:.2f}"
            )

    # Select best result
    index, best_result = max(enumerate(results_list), key=lambda x: x[1]["elbo"])
    if print_progress:
        print(f"Initialisation {index+1} has maximal ELBO and is returned")

    return best_result  # Note this return differs from return of cavi_multi_init_cpp as here return a dictionary for the result["parameters"], albeit for analysis purpose very similar