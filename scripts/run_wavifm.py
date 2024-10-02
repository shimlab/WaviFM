import concurrent.futures
from scripts.cavi import *
import pywt
import pandas as pd
import numpy as np

def run_wavifm(result_rna, n_length_scales, n_factors, n_x_indices, n_y_indices,
               max_iterations, relative_elbo_threshold, n_init, priors=None):
    
    ## Extracting Feature Matrices
    feats = np.vstack(result_rna['feature'].to_numpy())
    feats_colnames = [f'feat_{i+1}' for i in range(feats.shape[1])]
    df_feats = pd.DataFrame(feats, columns=feats_colnames)
    rna_feats = pd.concat([result_rna, df_feats], axis=1)

    ## Extract 2D matrices for each of the PCA factors (multithreading used for efficiency)
    feats_matrices = {}
    def get_feat_matrix(feats_name):
        new_feats_matrix = np.zeros((n_y_indices, n_x_indices))
        x_index_list = rna_feats['x_index'].to_list()
        y_index_list = rna_feats['y_index'].to_list()
        feats_value_list = rna_feats[feats_name]
        for x_index, y_index, feats_value in zip(x_index_list, y_index_list, feats_value_list):
            new_feats_matrix[y_index][x_index] = feats_value
        feats_matrices[feats_name] = new_feats_matrix
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_dict = {executor.submit(get_feat_matrix, feats_name): feats_name for feats_name in feats_colnames}
    
    ## Applying wavelet transformation to the feature matrices and filtering
    flattened_wavelet_matrices = []
    coeffs_matrices = []
    for feat_name in feats_colnames:
        coeffs = pywt.wavedec2(feats_matrices[feat_name], 'haar', level=n_length_scales)
        flattened_coeffs = []
        flattened_coeffs.append((coeffs[0].flatten(),))
        for res in coeffs[1:]:
            flattened_coeffs.append(tuple(matrix.flatten() for matrix in res))   
            
        flattened_wavelet_matrices.append(flattened_coeffs)
        coeffs_matrices.append(coeffs)
    
    # Setting dimension constants
    true_Y = flattened_wavelet_matrices
    n_resolutions = len(true_Y[0])
    n_features = len(true_Y)
    n_spots = n_x_indices*n_y_indices
    L_shape = get_L_shape(n_spots, n_resolutions, n_factors)
    Y_shape = get_Y_shape(n_spots, n_resolutions, n_features)
    p_pi_shape = (n_resolutions,)
    p_eta_shape = (n_factors,)
    ab_t_shape = (n_resolutions, n_factors)
    ab_tau_shape = (n_resolutions, n_features)
    F_shape = (n_factors, n_features)
    
    dimensions = {
        "n_factors" : n_factors,
        "n_resolutions" : n_resolutions,
        "n_features" : n_features,
        "n_spots" : n_spots,
        "L_shape" : L_shape,
        "Y_shape" : Y_shape,
        "p_pi_shape" : p_pi_shape,
        "p_eta_shape" : p_eta_shape,
        "ab_t_shape" : ab_t_shape,
        "ab_tau_shape" : ab_tau_shape,
        "F_shape" : F_shape
    }

    assert n_resolutions == n_length_scales + 1
    assert n_spots >= 4 ** (n_resolutions - 1)
    assert is_power_of_4(n_spots)
    
    # Run CAVI
    results = cavi_multi_init_cpp(
        true_Y,
        dimensions,
        max_iterations=max_iterations,
        relative_elbo_threshold=relative_elbo_threshold,
        n_init=n_init,
        priors=priors
    )
    return results