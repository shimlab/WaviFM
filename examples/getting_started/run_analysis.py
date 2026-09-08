"""
WaveFactor Getting Started: Self-Contained Spatial Factor Modeling Case Study.

This script simulates a 16x16 spatial transcriptomics dataset with known
ground-truth factor-gene relationships, fits the WaveFactor model, and verifies
the posterior factor activities and gene loadings against the ground truth.

Ground-Truth Simulation Setup:
  - Spatial Grid : 16x16 dyadic square lattice (256 spots)
  - Latent Factors: 2 spatial domains
      * Factor 1: Left domain (x < 8)
      * Factor 2: Right domain (x >= 8)
  - Gene Programs: 100 genes total
      * Genes 001-040: Active in Factor 1 (true loading = 1.5)
      * Genes 041-080: Active in Factor 2 (true loading = 1.5)
      * Genes 081-100: Unassociated background noise (true loading = 0.0)
"""

import os
import sys
from typing import Tuple, List
import numpy as np

# Ensure repository root is on Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wavefactor import WaveFactor


def simulate_spatial_data(
    side: int = 16,
    n_genes: int = 100,
    n_factors: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Simulates a synthetic spatial transcriptomics dataset with known ground truth.

    Parameters
    ----------
    side : int, default=16
        Lattice side length (grid has side x side spots).
    n_genes : int, default=100
        Total number of genes.
    n_factors : int, default=2
        Number of latent spatial factors.
    seed : int, default=42
        Random seed for deterministic generation.

    Returns
    -------
    expr : np.ndarray of shape (N_spots, n_genes)
        Simulated gene expression matrix with observation noise.
    coords : np.ndarray of shape (N_spots, 2)
        Spatial (x, y) integer lattice coordinates.
    gene_names : list of str
        Identifiers for each gene.
    L_true : np.ndarray of shape (N_spots, n_factors)
        Ground-truth spot-space spatial factor activities.
    F_true : np.ndarray of shape (n_factors, n_genes)
        Ground-truth factor-to-gene loadings matrix.
    """
    rng = np.random.RandomState(seed)
    n_spots = side * side

    # 1. 2D regular lattice coordinates (row-major: y = row, x = col)
    x = np.tile(np.arange(side), side)
    y = np.repeat(np.arange(side), side)
    coords = np.column_stack([x, y])

    # 2. Ground-truth spatial factor activities (L in R^(N x K))
    # Factor 1: Left domain (x < 8)
    # Factor 2: Right domain (x >= 8)
    L_true = np.zeros((n_spots, n_factors), dtype=np.float64)
    L_true[x < side // 2, 0] = 2.0
    L_true[x >= side // 2, 1] = 2.0

    # 3. Ground-truth factor loadings (F in R^(K x G))
    # 40 genes for Factor 1, 40 genes for Factor 2, 20 noise genes
    F_true = np.zeros((n_factors, n_genes), dtype=np.float64)
    F_true[0, 0:40] = 1.5
    F_true[1, 40:80] = 1.5

    # 4. Generate expression: signal (L @ F) + observation noise
    signal = L_true @ F_true
    noise = rng.normal(0, 0.4, size=signal.shape)
    expr = signal + noise

    # Weaker background expression for unassociated noise genes (indices 80-99)
    expr[:, 80:] = rng.normal(0, 0.2, size=(n_spots, 20))

    gene_names = [f"Gene_{i+1:03d}" for i in range(n_genes)]
    return expr, coords, gene_names, L_true, F_true


def main():
    print("=" * 68)
    print("           WaveFactor v2.0 - Getting Started Case Study             ")
    print("=" * 68)

    # 1. Simulate data directly in code
    expr, coords, gene_names, L_true, F_true = simulate_spatial_data(
        side=16,
        n_genes=100,
        n_factors=2,
        seed=42,
    )
    print("Simulated Data: 256 spots (16x16 grid), 100 genes, 2 spatial factors\n")

    # 2. Configure and fit WaveFactor estimator with canonical defaults
    print("Fitting WaveFactor estimator (K=2 factors, D=2 wavelet detail levels)...")
    model = WaveFactor(
        n_factors=2,
        n_length_scales=2,
        max_iter=500,
        tol=1e-4,
        n_init=3,
        random_state=42,
        verbose=False,
    )
    model.fit(expr, coords)
    result = model.get_result()

    # 3. Convergence summary
    print(f"\nConvergence:\n  Final ELBO: {model.elbo_:.2f} ({result.n_iter} iterations)")

    # 4. Spatial factor domain means
    print("\nSpatial Factor Domain Means:")
    for k in range(model.n_factors):
        factor_map = result.spatial_factor_maps[:, :, k]
        left_mean = factor_map[:, :8].mean()
        right_mean = factor_map[:, 8:].mean()
        print(f"  Factor {k+1}: Left = {left_mean:.2f}, Right = {right_mean:.2f}")

    # 5. Gene selection evaluated against ground truth
    pips = model.gene_pip_
    print("\nActive Genes (Posterior Inclusion Probability > 0.5):")
    for k in range(model.n_factors):
        active_indices = np.where(pips[k] > 0.5)[0]
        f1_cnt = np.sum(active_indices < 40)
        f2_cnt = np.sum((active_indices >= 40) & (active_indices < 80))
        noise_cnt = np.sum(active_indices >= 80)
        print(f"  Factor {k+1}:")
        print(f"    - True Program 1 (Genes 001-040): {f1_cnt}/40")
        print(f"    - True Program 2 (Genes 041-080): {f2_cnt}/40")
        print(f"    - Background Noise (Genes 081-100): {noise_cnt}/20 false positives")

    print("=" * 68)


if __name__ == "__main__":
    main()
