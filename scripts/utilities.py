# This file contains various utilities that are not directly used by CAVI, but instead used for other purposes like data preprocessing, simulations, data format transformation, etc.

import numpy as np
import math

# Utilities pertaining to generating values with specific shape or purpose in CAVI algorithm simulation/initialisation/etc.


def L_shaped_pi(p_pi, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]
    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            p_pi_i = p_pi[i]
            for j in range(len(L_shape[l][i])):
                matrix_shape = L_shape[l][i][j].shape
                lvl_matrices.append(np.random.binomial(1, p_pi_i, size=matrix_shape))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def L_shaped_pi_organised(p_pi, dimensions):
    # Similar to L_shaped_pi, but instead of simulating using binomial distribution, p_pi is just used to deterministically set values of pi in an ordered manner too for simplicity
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]
    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            p_pi_i = p_pi[i]
            for j in range(len(L_shape[l][i])):
                max_k = len(L_shape[l][i][j])
                matrix = np.zeros_like(L_shape[l][i][j]).astype(np.float64)
                n_ones = np.floor(max_k * p_pi_i).astype(int)
                for k in range(n_ones):
                    matrix[k] = 1.0
                lvl_matrices.append(matrix)
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def L_rand_norm_from_t(t, pi, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]
    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            t_i_l = t[i][l]
            for j in range(len(L_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(L_shape[l][i][j])):
                    pi_ijk_l = pi[l][i][j][k]
                    wavelet_matrix.append(
                        pi_ijk_l * np.random.normal(0, np.sqrt(1 / t_i_l))
                    )
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def L_rand_norm(alpha_t, beta_t, pi, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]
    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            alpha_t_i_l = alpha_t[i][l]
            beta_t_i_l = beta_t[i][l]
            t_i_l = np.random.gamma(alpha_t_i_l, 1 / beta_t_i_l)
            for j in range(len(L_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(L_shape[l][i][j])):
                    pi_ijk_l = pi[l][i][j][k]
                    wavelet_matrix.append(
                        pi_ijk_l * np.random.normal(0, np.sqrt(1 / t_i_l))
                    )
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def F_rand_norm(eta, dimensions):
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    values = []
    for i in range(n_factors):
        factor_values = []
        for j in range(n_features):
            eta_i_j = eta[i][j]
            factor_values.append(eta_i_j * np.random.normal(0, 1))
        values.append(factor_values)
    return np.array(values)


def LF_product(L, F, dimensions):
    # Multiplies L,F to obtain a Y=LF object with format akin to Y objects in CAVI
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    n_resolutions = dimensions["n_resolutions"]
    Y_shape = dimensions["Y_shape"]
    values = []
    for l in range(n_features):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            for j in range(len(Y_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(Y_shape[l][i][j])):
                    mu = 0
                    for m in range(
                        n_factors
                    ):  # Perform matrix product to get mu for Y_ijk_l
                        L_ijk_m = L[m][i][j][k]
                        F_m_l = F[m][l]
                        mu += L_ijk_m * F_m_l
                    wavelet_matrix.append(mu)
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def Y_rand_norm(L, F, alpha_tau, beta_tau, dimensions):
    mu_values = LF_product(L, F, dimensions)

    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    n_resolutions = dimensions["n_resolutions"]
    Y_shape = dimensions["Y_shape"]
    values = []
    for l in range(n_features):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            alpha_tau_i_l = alpha_tau[i][l]
            beta_tau_i_l = beta_tau[i][l]
            tau_i_l = np.random.gamma(alpha_tau_i_l, 1 / beta_tau_i_l)
            for j in range(len(Y_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(Y_shape[l][i][j])):
                    mu_ijk_l = mu_values[l][i][j][k]
                    wavelet_matrix.append(
                        np.random.normal(mu_ijk_l, np.sqrt(1 / tau_i_l))
                    )
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def Y_variances(L, F, dimensions):
    # Computes a variance numpy matrix with same shape as tau_i_l matrix to serve for purposes of computing tau_i_l alongside signal to noise ratios for sake of simulation
    mu_values = LF_product(L, F, dimensions)

    n_features = dimensions["n_features"]
    n_resolutions = dimensions["n_resolutions"]
    Y_shape = dimensions["Y_shape"]
    values = np.zeros((n_resolutions, n_features))
    for l in range(n_features):
        for i in range(n_resolutions):
            resolution_means = []
            for j in range(len(Y_shape[l][i])):
                for k in range(len(Y_shape[l][i][j])):
                    mu_ijk_l = mu_values[l][i][j][k]
                    resolution_means.append(mu_ijk_l)
            values[i][l] = np.var(resolution_means, ddof=0)
    return values


def Y_rand_norm_from_tau(L, F, tau, dimensions):
    # Similar to Y_rand_norm but with tau_i_l provided as opposed to simulated
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    n_resolutions = dimensions["n_resolutions"]
    Y_shape = dimensions["Y_shape"]
    values = []
    for l in range(n_features):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            tau_i_l = tau[i][l]
            for j in range(len(Y_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(Y_shape[l][i][j])):
                    mu = 0
                    for m in range(
                        n_factors
                    ):  # Perform matrix product to get mu for Y_ijk_l
                        L_ijk_m = L[m][i][j][k]
                        F_m_l = F[m][l]
                        mu += L_ijk_m * F_m_l
                    if np.isinf(tau_i_l):
                        wavelet_matrix.append(np.random.normal(mu, 0))
                    else:
                        wavelet_matrix.append(
                            np.random.normal(mu, np.sqrt(1 / tau_i_l))
                        )
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def Y_noiseless(L, F, dimensions):
    # Similar to Y_rand_norm but with noise as 0 (i.e. sets elements of Y as element of matrix product L,F directly without adding noise (that depends on tau_i_l))
    n_factors = dimensions["n_factors"]
    n_features = dimensions["n_features"]
    n_resolutions = dimensions["n_resolutions"]
    Y_shape = dimensions["Y_shape"]
    values = []
    for l in range(n_features):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            for j in range(len(Y_shape[l][i])):
                wavelet_matrix = []
                for k in range(len(Y_shape[l][i][j])):
                    mu = 0
                    for m in range(
                        n_factors
                    ):  # Perform matrix product to get mu for Y_ijk_l
                        L_ijk_m = L[m][i][j][k]
                        F_m_l = F[m][l]
                        mu += L_ijk_m * F_m_l
                    wavelet_matrix.append(
                        mu
                    )  # Only difference to Y_rand_norm is this line
                lvl_matrices.append(np.array(wavelet_matrix))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def L_shaped_rand(low, high, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    Y_shape = dimensions["Y_shape"]

    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            for j in range(len(Y_shape[l][i])):
                matrix_shape = Y_shape[l][i][j].shape
                lvl_matrices.append(np.random.uniform(low, high, size=matrix_shape))
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


def L_shaped_log_rand(low, high, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    Y_shape = dimensions["Y_shape"]

    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            lvl_matrices = []
            for j in range(len(Y_shape[l][i])):
                matrix_shape = Y_shape[l][i][j].shape
                lvl_matrices.append(
                    np.log(np.random.uniform(low, high, size=matrix_shape))
                )
            factor_vals.append(lvl_matrices)
        values.append(factor_vals)
    return values


# Utilities to reshape the ijk (resolution, wavelet type, wavelet index) indexing of wavelet coefficient quantities into a single index (to facilitate other operations with them)
def flatten_data(data):
    # Flattens data indexed by l,i,j,k where i,j,k are wavelet coefficient indices. Collapses into a single index
    flattened_data = []
    index_map = []  # To store the original indices

    for l, l_data in enumerate(data):
        for i, i_data in enumerate(l_data):
            for j, j_data in enumerate(i_data):
                for k, value in enumerate(j_data):
                    flattened_data.append(value)
                    index_map.append((l, i, j, k))

    flattened_data = np.array(flattened_data)
    return {"index_map": index_map, "flattened_data": flattened_data}


def get_wavelet_indices(data):
    # Returns the set of wavelet coefficient indices
    index_map = []  # To store the original indices
    for l, l_data in enumerate(data):
        for i, i_data in enumerate(l_data):
            for j, j_data in enumerate(i_data):
                for k, value in enumerate(j_data):
                    index_map.append((l, i, j, k))
    indices = list({(i, j, k) for _, i, j, k in index_map})
    indices.sort()
    return indices


def flatten_data_to_2D(data):
    # Flattens data indexed by l,i,j,k where i,j,k are wavelet coefficient indices. Collapses into a two indices, one being l, other being i,j,k collapsed
    # To convert into a 2D array where rows are (i, j, k) combinations and columns are l
    unique_indices = get_wavelet_indices(data)

    # Initialize a matrix with dimensions (num_data_points, num_variables)
    num_variables = len(data)
    num_data_points = len(unique_indices)
    data_matrix = np.zeros((num_data_points, num_variables))

    # Fill the data matrix
    for idx, (i, j, k) in enumerate(unique_indices):
        for l in range(num_variables):
            data_matrix[idx, l] = data[l][i][j][k]

    return data_matrix


def reshape_flattened_2D(flattened_2D, wavelet_indices):
    # Reshapes flattened 2D data to original l,i,j,k indexed form
    reshaped_data = []
    for l, l_data in enumerate(flattened_2D):
        if l not in reshaped_data:
            reshaped_data.append([])
        for idx, (i, j, k) in enumerate(wavelet_indices):
            # Initialize nested dictionary structure if not already done
            if len(reshaped_data[l]) <= i:
                reshaped_data[l].append([])
            if len(reshaped_data[l][i]) <= j:
                reshaped_data[l][i].append([])
            if len(reshaped_data[l][i][j]) <= k:
                reshaped_data[l][i][j].append([])

            # Append the principal components for each (i, j, k)
            reshaped_data[l][i][j][k] = l_data[idx]

    return reshaped_data


def flatten_L_shaped_values(L_shaped_values, dimensions):
    # Flattens L_shaped values to a 2D numpy array with shape (n_coefficients, n_factors) for each of the resolutions
    # that is, the wavelet coefficients indices (3D in our setup) is collapsed to being from a 1D index set
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]

    values = []
    for l in range(n_factors):
        factor_vals = []
        for i in range(n_resolutions):
            for j in range(len(L_shape[l][i])):
                for k in range(len(L_shape[l][i][j])):
                    factor_vals.append(L_shaped_values[l][i][j][k])
        values.append(factor_vals)
    return np.array(values).transpose()


def flatten_L_shaped_values_by_resolution(L_shaped_values, dimensions):
    n_resolutions = dimensions["n_resolutions"]
    n_factors = dimensions["n_factors"]
    L_shape = dimensions["L_shape"]

    values = []
    for i in range(n_resolutions):
        resolution_values = []
        for l in range(n_factors):
            factor_vals = []
            for j in range(len(L_shape[l][i])):
                for k in range(len(L_shape[l][i][j])):
                    factor_vals.append(L_shaped_values[l][i][j][k])
            resolution_values.append(factor_vals)
        values.append(np.array(resolution_values).transpose())
    return values


# Utilities for other miscellaneous purposes


def threshold_to_0_or_1(array, upper_threshold, lower_threshold):
    # Given a 2D numpy array, its values above upper threshold rounded to 1, those below lower_threshold rounded to 0
    upper_rounded = np.where(array > upper_threshold, 1, array)
    upper_and_lower_rounded = np.where(
        upper_rounded < lower_threshold, 0, upper_rounded
    )
    return upper_and_lower_rounded


def n_parameters(dimensions):
    # Returns the number of dimensions fitted using variational inference
    K = dimensions["n_factors"]
    G = dimensions["n_features"]
    R = dimensions["n_resolutions"]
    N = dimensions["n_spots"]
    return 3 * N * K + 3 * K * G + 2 * R * G + 2 * R * K


def log_4(x):
    return math.log10(x) / math.log10(4)


def get_tau(snr, Y_variances):
    # This is for use in SNR in wavelet space Y mock data generation
    tau = np.empty_like(Y_variances)
    Y_variances_shape = Y_variances.shape
    for i in range(Y_variances_shape[0]):
        for l in range(Y_variances_shape[1]):
            tau[i][l] = snr / Y_variances[i][l] if Y_variances[i][l] != 0 else np.inf
    return tau


def get_tau_spot(snr, Y):
    # This is for use in SNR in spot space Y mock data generation
    # Assumes the input Y is indexed by i,j where i is spot index, j is feature index
    tau = np.empty_like(Y)
    Y_variances = np.var(Y, axis=0, ddof=0)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            tau[i][j] = snr / Y_variances[j] if Y_variances[j] != 0 else np.inf
    return tau


def get_error_from_tau_spot(tau):
    # This is for use in SNR in spot space Y mock data generation
    # Assumes the input tau is indexed by i,j where i is spot index, j is feature index
    error = np.empty_like(tau)
    for i in range(tau.shape[0]):
        for j in range(tau.shape[1]):
            tau_i_j = tau[i][j]
            if np.isinf(tau_i_j):
                error[i][j] = np.random.normal(0, 0)
            else:
                error[i][j] = np.random.normal(loc=0, scale=np.sqrt(1 / tau_i_j))
    return error


def is_power_of_4(n):
    return math.ceil(log_4(n)) == math.floor(log_4(n))


def get_L_shape(n_spots, n_resolutions, n_factors):
    return [
        [[np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - 1)))))]]
        + [
            [
                np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - i - 1)))))
                for j in range(3)
            ]
            for i in range(n_resolutions - 1)
        ]
        for l in range(n_factors)
    ]


def get_Y_shape(n_spots, n_resolutions, n_features):
    return [
        [[np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - 1)))))]]
        + [
            [
                np.array([0.0] * (int(n_spots / (4 ** (n_resolutions - i - 1)))))
                for j in range(3)
            ]
            for i in range(n_resolutions - 1)
        ]
        for l in range(n_features)
    ]


class AttributeIndexer:
    def __init__(self, obj):
        self._obj = obj

    def __getitem__(self, key):
        return getattr(self._obj, key)

    def __setitem__(self, key, value):
        setattr(self._obj, key, value)

    def __delitem__(self, key):
        delattr(self._obj, key)
