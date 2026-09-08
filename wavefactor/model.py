"""
Scikit-learn compatible Estimator for WaveFactor.
"""

from typing import Optional, Union, Tuple, List, Dict, Any
import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(RuntimeError):
        pass

    # Minimal fallback base classes if scikit-learn is not installed
    class BaseEstimator:
        def get_params(self, deep=True):
            return {
                k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and not k.endswith("_")
            }
        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y=y, **fit_params).transform(X)


from .priors import Priors
from .data import prepare_spatial_data, WaveFactorData
from .results import WaveFactorResult
from .engine import run_cavi


class WaveFactor(BaseEstimator):
    """
    WaveFactor: Bayesian Multiresolution Wavelet Spatial Factor Model.

    Parameters
    ----------
    n_factors : int, default=10
        Number of latent spatial factors (K).
    n_length_scales : int, default=4
        Number of wavelet detail length scales / decomposition levels (D).
        Total resolutions R = D + 1.
    spatial_prior : str, float, list, or np.ndarray, default="decay"
        Prior inclusion probabilities for spatial wavelet coefficients (pi).
        Can be a preset ("decay", "sparse", "uniform"), scalar probability in (0, 1),
        or explicit array of length R.
    gene_prior : float, list, or np.ndarray, default=0.2
        Prior inclusion probability for gene loadings (eta).
    precision_prior_t : Tuple[float, float], default=(1.0, 1.0)
        Shape (alpha) and rate (beta) for factor precision Gamma prior.
    precision_prior_tau : Tuple[float, float], default=(1.0, 1.0)
        Shape (alpha) and rate (beta) for noise precision Gamma prior.
    grid_side_length : int, optional
        Side length L of the square L x L spatial grid (must be a power of 2,
        e.g. 16, 32, 64, 128). Total grid cells = L * L. If None, automatically
        computed as the smallest power of 2 that encloses the spatial coordinates.
    max_iter : int, default=1000
        Maximum CAVI optimization iterations per start.
    tol : float, default=1e-5
        Relative ELBO change convergence threshold (|Delta ELBO / ELBO| < tol).
    n_init : int, default=5
        Number of randomized initial parameter starts.
    n_jobs : int, default=-1
        Number of parallel worker processes (-1 uses all available CPU cores).
    random_state : int, optional
        Random seed for deterministic initialization and reproducibility.
    verbose : bool, default=True
        Whether to log progress during CAVI optimization.
    """

    def __init__(
        self,
        n_factors: int = 10,
        n_length_scales: int = 4,
        spatial_prior: Optional[np.ndarray] = None,
        gene_prior: Optional[np.ndarray] = None,
        alpha_t: Optional[np.ndarray] = None,
        beta_t: Optional[np.ndarray] = None,
        alpha_tau: Optional[np.ndarray] = None,
        beta_tau: Optional[np.ndarray] = None,
        max_iter: int = 1000,
        tol: float = 1e-5,
        n_init: int = 5,
        n_jobs: int = -1,
        random_state: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_factors = n_factors
        self.n_length_scales = n_length_scales
        self.spatial_prior = spatial_prior
        self.gene_prior = gene_prior
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.alpha_tau = alpha_tau
        self.beta_tau = beta_tau
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes (populated after fit)
        self.result_: Optional[WaveFactorResult] = None
        self.data_: Optional[WaveFactorData] = None

    def fit(
        self,
        X: np.ndarray,
        coords: np.ndarray,
    ) -> "WaveFactor":
        """
        Fits the WaveFactor model on spatial transcriptomics expression data.

        Parameters
        ----------
        X : np.ndarray
            Gene expression data matrix of shape (N_spots, N_features).
            N_spots must be an exact power of 4 (e.g. 64, 256, 1024, 4096).
        coords : np.ndarray
            Spatial (x, y) integer grid coordinates for each spot, shape (N_spots, 2).
            Must form a unique 1-to-1 mapping onto the [0, L-1] x [0, L-1] lattice.

        Returns
        -------
        self : WaveFactor
            Fitted estimator.
        """
        # 1. Ingest and preprocess spatial data (strict grid validation and 2D DWT)
        self.data_ = prepare_spatial_data(
            X=X,
            coords=coords,
            n_factors=self.n_factors,
            n_length_scales=self.n_length_scales,
        )

        # 2. Build prior hyperparameter matrices
        priors_obj = Priors(
            spatial_prior=self.spatial_prior,
            gene_prior=self.gene_prior,
            alpha_t=self.alpha_t,
            beta_t=self.beta_t,
            alpha_tau=self.alpha_tau,
            beta_tau=self.beta_tau,
        )
        priors_dict = priors_obj.build_priors_dict(self.data_.dimensions)

        # 3. Execute Coordinate Ascent Variational Inference
        best_cavi_out = run_cavi(
            true_Y=self.data_.true_Y,
            dimensions=self.data_.dimensions,
            priors=priors_dict,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=self.n_init,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # 4. Construct rich result object
        self.result_ = WaveFactorResult(
            parameters=best_cavi_out["parameters"],
            elbo_record=best_cavi_out["elbo_record"],
            elbo=best_cavi_out["elbo"],
            dimensions=self.data_.dimensions,
            grid_side_length=self.data_.grid_side_length,
            spot_grid_coords=self.data_.spot_grid_coords,
            coords=self.data_.coords,
            cpp_time=best_cavi_out.get("cpp_time", 0.0),
        )

        return self

    def transform(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the inferred spot-space spatial factor activities.

        Parameters
        ----------
        X : Optional[np.ndarray], default=None
            Ignored. Present for scikit-learn TransformerMixin API compatibility.

        Returns
        -------
        np.ndarray
            Spatial factors matrix of shape (N_spots, K).
        """
        if self.result_ is None:
            raise RuntimeError("This WaveFactor instance is not fitted yet. Call 'fit' before 'transform'.")
        return self.result_.factors

    def fit_transform(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        **fit_params,
    ) -> np.ndarray:
        """
        Fits the model and returns the inferred spot-space spatial factors.

        Parameters
        ----------
        X : np.ndarray
            Gene expression data matrix of shape (N_spots, N_features).
        coords : np.ndarray
            Spatial (x, y) integer grid coordinates, shape (N_spots, 2).

        Returns
        -------
        np.ndarray
            Spatial factors matrix S of shape (N_spots, K).
        """
        return self.fit(X, coords).transform(X)

    def get_result(self) -> WaveFactorResult:
        """Returns the rich WaveFactorResult object."""
        if self.result_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
        return self.result_

    @property
    def factors_(self) -> np.ndarray:
        """Spot-space spatial factor activities S in R^(N_spots x K)."""
        return self.get_result().factors

    @property
    def loadings_(self) -> np.ndarray:
        """Factor-to-gene loadings matrix F in R^(K x G)."""
        return self.get_result().loadings

    @property
    def elbo_(self) -> float:
        """Final converged Evidence Lower Bound (ELBO)."""
        return self.get_result().elbo

    @property
    def elbo_history_(self) -> np.ndarray:
        """ELBO convergence trace across iterations."""
        return self.get_result().elbo_history

    @property
    def spatial_pip_(self) -> List:
        """Posterior inclusion probabilities for spatial wavelet coefficients (r_pi)."""
        return self.get_result().spatial_pip

    @property
    def gene_pip_(self) -> np.ndarray:
        """Posterior inclusion probabilities for gene loadings (r_eta)."""
        return self.get_result().gene_pip

    @property
    def precision_t_(self) -> np.ndarray:
        """Posterior factor precision expectation E[t]."""
        return self.get_result().precision_t

    @property
    def precision_tau_(self) -> np.ndarray:
        """Posterior noise precision expectation E[tau]."""
        return self.get_result().precision_tau
