"""
Results container and automated 2D Inverse Wavelet Transform (IDWT) for WaveFactor.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pywt


class WaveFactorResult:
    """
    Container holding WaveFactor posterior parameter estimates, convergence history,
    and automated 2D Inverse Discrete Wavelet Transform (IDWT) reconstructions.

    Attributes
    ----------
    parameters : Dict[str, Any]
        Raw CAVI parameter dictionary returned by the C++ engine.
    elbo_record : np.ndarray
        ELBO trajectory values across CAVI iterations.
    elbo : float
        Final converged Evidence Lower Bound.
    dimensions : Dict[str, Any]
        Dictionary with keys 'n_spots', 'n_resolutions', 'n_factors', 'n_features'.
    grid_side_length : int
        Side length L of the square L x L spatial grid.
    grid_shape : Tuple[int, int]
        2D grid dimensions (grid_side_length, grid_side_length).
    spot_grid_coords : np.ndarray
        Array of shape (N_spots, 2) holding Cartesian [grid_x, grid_y] coordinates.
    grid_x : np.ndarray
        1D integer grid x-coordinates (columns in 0..L-1).
    grid_y : np.ndarray
        1D integer grid y-coordinates (rows in 0..L-1).
    coords : np.ndarray, optional
        Original spot spatial coordinates (N_spots, 2).
    cpp_time : float
        Elapsed C++ CAVI runtime in seconds.
    """

    def __init__(
        self,
        parameters: Dict[str, Any],
        elbo_record: List[float],
        elbo: float,
        dimensions: Dict[str, Any],
        grid_side_length: int,
        spot_grid_coords: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None,
        cpp_time: float = 0.0,
    ):
        self.parameters = parameters
        self.elbo_record = np.asarray(elbo_record, dtype=np.float64)
        self.elbo = float(elbo)
        self.dimensions = dimensions
        self.grid_side_length = int(grid_side_length)
        self.spot_grid_coords = spot_grid_coords
        self.coords = coords
        self.cpp_time = cpp_time

        # Cached computed properties
        self._spatial_factor_maps: Optional[np.ndarray] = None
        self._spot_factors: Optional[np.ndarray] = None
        self._gene_loadings: Optional[np.ndarray] = None

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """2D spatial grid shape (grid_side_length, grid_side_length)."""
        return (self.grid_side_length, self.grid_side_length)

    @property
    def grid_x(self) -> Optional[np.ndarray]:
        """1D integer grid x-coordinates (columns in 0..W-1)."""
        return self.spot_grid_coords[:, 0] if self.spot_grid_coords is not None else None

    @property
    def grid_y(self) -> Optional[np.ndarray]:
        """1D integer grid y-coordinates (rows in 0..H-1)."""
        return self.spot_grid_coords[:, 1] if self.spot_grid_coords is not None else None

    @property
    def n_iter(self) -> int:
        """Number of completed CAVI iterations."""
        return max(len(self.elbo_record) - 1, 0)

    @property
    def elbo_history(self) -> np.ndarray:
        """Trace of Evidence Lower Bound values across iterations."""
        return self.elbo_record

    @property
    def n_factors(self) -> int:
        """Number of latent factors (K)."""
        return self.dimensions["n_factors"]

    @property
    def n_features(self) -> int:
        """Number of genes / features (G)."""
        return self.dimensions["n_features"]

    @property
    def n_resolutions(self) -> int:
        """Number of wavelet resolution levels (R)."""
        return self.dimensions["n_resolutions"]

    @property
    def spatial_factor_maps(self) -> np.ndarray:
        """
        2D Continuous Spatial Factor Activity Maps S in R^(N_y x N_x x K),
        reconstructed via automated 2D Inverse Wavelet Transform (IDWT).
        """
        if self._spatial_factor_maps is None:
            self._reconstruct_spatial_factors()
        return self._spatial_factor_maps

    @property
    def factors(self) -> np.ndarray:
        """
        Spot-space latent factor activities S in R^(N_spots x K).
        If continuous coordinates were provided, factors are evaluated at original spot locations.
        """
        if self._spot_factors is None:
            self._reconstruct_spatial_factors()
        return self._spot_factors

    @property
    def loadings(self) -> np.ndarray:
        """
        Factor-to-gene loadings matrix F in R^(K x G).
        Computed as the expected gene loading E[F] = mu_F * r_eta.
        """
        if self._gene_loadings is None:
            mu_F = np.asarray(self.parameters["mu_F"], dtype=np.float64)
            r_eta = np.exp(np.asarray(self.parameters["log_r_eta"], dtype=np.float64))
            self._gene_loadings = mu_F * r_eta
        return self._gene_loadings

    @property
    def mu_F(self) -> np.ndarray:
        """Gaussian slab variational mean matrix for gene loadings mu_F in R^(K x G)."""
        return np.asarray(self.parameters["mu_F"], dtype=np.float64)

    @property
    def sigma_squared_F(self) -> np.ndarray:
        """Gaussian slab variational variance matrix for gene loadings sigma_F^2 in R^(K x G)."""
        return np.asarray(self.parameters["sigma_squared_F"], dtype=np.float64)

    @property
    def loadings_variance(self) -> np.ndarray:
        """
        Analytical posterior variance of gene loadings Var_q(F) in R^(K x G).
        Var_q(F) = r_eta * sigma_F^2 + r_eta * (1 - r_eta) * mu_F^2.
        """
        r = self.gene_pip
        s2 = self.sigma_squared_F
        m = self.mu_F
        return r * s2 + r * (1.0 - r) * (m ** 2)

    @property
    def mu_L(self) -> List:
        """Gaussian slab variational mean tensor for spatial wavelet loadings mu_L."""
        return self.parameters["mu_L"]

    @property
    def sigma_squared_L(self) -> List:
        """Gaussian slab variational variance tensor for spatial wavelet loadings sigma_L^2."""
        return self.parameters["sigma_squared_L"]

    @property
    def gene_pip(self) -> np.ndarray:
        """Gene posterior inclusion probabilities r_eta in R^(K x G)."""
        return np.exp(np.asarray(self.parameters["log_r_eta"], dtype=np.float64))

    @property
    def spatial_pip(self) -> List:
        """Spatial wavelet posterior inclusion probabilities r_pi (nested resolution lists)."""
        log_r_pi = self.parameters["log_r_pi"]
        return self._apply_exp_nested(log_r_pi)

    @property
    def alpha_hat_t(self) -> np.ndarray:
        """Variational Gamma shape parameter alpha_hat_t for factor precision in R^(R x K)."""
        return np.asarray(self.parameters["alpha_hat_t"], dtype=np.float64)

    @property
    def beta_hat_t(self) -> np.ndarray:
        """Variational Gamma rate parameter beta_hat_t for factor precision in R^(R x K)."""
        return np.asarray(self.parameters["beta_hat_t"], dtype=np.float64)

    @property
    def precision_t(self) -> np.ndarray:
        """Posterior expected factor precision E[t] = alpha_hat_t / beta_hat_t in R^(R x K)."""
        return self.alpha_hat_t / self.beta_hat_t

    @property
    def alpha_hat_tau(self) -> np.ndarray:
        """Variational Gamma shape parameter alpha_hat_tau for noise precision in R^(R x G)."""
        return np.asarray(self.parameters["alpha_hat_tau"], dtype=np.float64)

    @property
    def beta_hat_tau(self) -> np.ndarray:
        """Variational Gamma rate parameter beta_hat_tau for noise precision in R^(R x G)."""
        return np.asarray(self.parameters["beta_hat_tau"], dtype=np.float64)

    @property
    def precision_tau(self) -> np.ndarray:
        """Posterior expected noise precision E[tau] = alpha_hat_tau / beta_hat_tau in R^(R x G)."""
        return self.alpha_hat_tau / self.beta_hat_tau

    def _reconstruct_spatial_factors(self):
        """Executes 2D IDWT (pywt.waverec2) across all factors."""
        K = self.n_factors
        R = self.n_resolutions
        mu_L = self.parameters["mu_L"]
        log_r_pi = self.parameters["log_r_pi"]
        n_y, n_x = self.grid_shape

        factor_maps = np.zeros((n_y, n_x, K), dtype=np.float64)

        for l in range(K):
            coeffs = []
            # Approximation level 0 (1 matrix)
            approx_mu = np.asarray(mu_L[l][0][0], dtype=np.float64)
            approx_r = np.exp(np.asarray(log_r_pi[l][0][0], dtype=np.float64))
            approx_vals = approx_mu * approx_r
            side_0 = int(round(np.sqrt(len(approx_vals))))
            coeffs.append(approx_vals.reshape((side_0, side_0)))

            # Detail levels 1, ..., R-1 (3 matrices each: cH, cV, cD)
            for r in range(1, R):
                detail_tuple = []
                for j in range(3):
                    det_mu = np.asarray(mu_L[l][r][j], dtype=np.float64)
                    det_r = np.exp(np.asarray(log_r_pi[l][r][j], dtype=np.float64))
                    det_vals = det_mu * det_r
                    side_r = int(round(np.sqrt(len(det_vals))))
                    detail_tuple.append(det_vals.reshape((side_r, side_r)))
                coeffs.append(tuple(detail_tuple))

            # Perform 2D IDWT
            rec_map = pywt.waverec2(coeffs, "haar")
            # Ensure shape matches grid
            if rec_map.shape != (n_y, n_x):
                rec_map = rec_map[:n_y, :n_x]
            factor_maps[:, :, l] = rec_map

        self._spatial_factor_maps = factor_maps

        # Map to spot space if original spot grid coordinates exist
        if self.spot_grid_coords is not None:
            gx = self.spot_grid_coords[:, 0]  # horizontal column index
            gy = self.spot_grid_coords[:, 1]  # vertical row index
            self._spot_factors = factor_maps[gy, gx, :]
        else:
            self._spot_factors = factor_maps.reshape(-1, K)

    def _apply_exp_nested(self, obj: Any) -> Any:
        """Recursively applies exp() to nested list structures."""
        if isinstance(obj, list):
            return [self._apply_exp_nested(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return np.exp(obj)
        elif isinstance(obj, (int, float)):
            return np.exp(obj)
        return obj
