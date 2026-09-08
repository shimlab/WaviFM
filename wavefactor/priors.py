"""
Priors module for WaveFactor.

Provides flexible, canonical tensor prior specifications for Bayesian wavelet factor modeling.
Supports exact mathematical tensor dimensions matching C++ CAVI updates:
  - spatial_prior: 1D array of shape (n_resolutions,)
  - gene_prior: 2D array of shape (n_factors, n_features)
  - alpha_t, beta_t: 2D array of shape (n_resolutions, n_factors)
  - alpha_tau, beta_tau: 2D array of shape (n_resolutions, n_features)
"""

from typing import Optional, Dict, Any
import numpy as np


class Priors:
    """
    Container and builder for WaveFactor prior hyperparameters.

    Parameters
    ----------
    spatial_prior : np.ndarray, optional
        Prior inclusion probabilities for spatial wavelet coefficients (pi).
        Must be a 1D array of shape (n_resolutions,) with values in (0, 1).
        Default (None): uninformative flat baseline np.full(n_resolutions, 0.5).
    gene_prior : np.ndarray, optional
        Prior inclusion probabilities for gene loadings (eta).
        Must be a 2D array of shape (n_factors, n_features) with values in (0, 1).
        Default (None): standard baseline np.full((n_factors, n_features), 0.5).
    alpha_t : np.ndarray, optional
        Shape hyperparameter matrix for Gamma prior on factor precision t.
        Must be a 2D array of shape (n_resolutions, n_factors) with values > 0.
        Default (None): np.ones((n_resolutions, n_factors)).
    beta_t : np.ndarray, optional
        Rate hyperparameter matrix for Gamma prior on factor precision t.
        Must be a 2D array of shape (n_resolutions, n_factors) with values > 0.
        Default (None): np.ones((n_resolutions, n_factors)).
    alpha_tau : np.ndarray, optional
        Shape hyperparameter matrix for Gamma prior on noise precision tau.
        Must be a 2D array of shape (n_resolutions, n_features) with values > 0.
        Default (None): np.ones((n_resolutions, n_features)).
    beta_tau : np.ndarray, optional
        Rate hyperparameter matrix for Gamma prior on noise precision tau.
        Must be a 2D array of shape (n_resolutions, n_features) with values > 0.
        Default (None): np.ones((n_resolutions, n_features)).
    eps : float, default=1e-12
        Numerical guard to prevent log(0) evaluation.
    """

    def __init__(
        self,
        spatial_prior: Optional[np.ndarray] = None,
        gene_prior: Optional[np.ndarray] = None,
        alpha_t: Optional[np.ndarray] = None,
        beta_t: Optional[np.ndarray] = None,
        alpha_tau: Optional[np.ndarray] = None,
        beta_tau: Optional[np.ndarray] = None,
        eps: float = 1e-12,
    ):
        self.spatial_prior = spatial_prior
        self.gene_prior = gene_prior
        self.alpha_t = alpha_t
        self.beta_t = beta_t
        self.alpha_tau = alpha_tau
        self.beta_tau = beta_tau
        self.eps = eps

    def build_priors_dict(self, dimensions: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Constructs the log-prior arrays and Gamma hyperparameter matrices matching
        the exact C++ CAVI dimensional requirements.

        Parameters
        ----------
        dimensions : dict
            Dictionary containing 'n_resolutions', 'n_factors', 'n_features',
            'p_pi_shape', 'F_shape', 'ab_t_shape', 'ab_tau_shape'.

        Returns
        -------
        dict
            Dictionary with keys 'log_p_pi', 'log_p_eta', 'alpha_t', 'beta_t',
            'alpha_tau', 'beta_tau'.
        """
        n_resolutions = dimensions["n_resolutions"]
        n_factors = dimensions["n_factors"]
        n_features = dimensions["n_features"]
        ab_t_shape = dimensions["ab_t_shape"]
        ab_tau_shape = dimensions["ab_tau_shape"]

        # 1. Resolve spatial prior p_pi (1D array)
        if self.spatial_prior is not None:
            if not isinstance(self.spatial_prior, np.ndarray) or self.spatial_prior.ndim != 1:
                raise TypeError("spatial_prior must be a 1D numpy array of shape (n_resolutions,)")
            if len(self.spatial_prior) != n_resolutions:
                raise ValueError(
                    f"spatial_prior length ({len(self.spatial_prior)}) must match n_resolutions ({n_resolutions})"
                )
            if np.any(self.spatial_prior <= 0.0) or np.any(self.spatial_prior >= 1.0):
                raise ValueError("All spatial_prior probabilities must be strictly in (0, 1)")
            p_pi = self.spatial_prior.astype(np.float64)
        else:
            p_pi = np.full(n_resolutions, 0.5, dtype=np.float64)

        log_p_pi = np.log(np.clip(p_pi, self.eps, 1.0 - self.eps))

        # 2. Resolve gene prior p_eta (2D array)
        if self.gene_prior is not None:
            if not isinstance(self.gene_prior, np.ndarray) or self.gene_prior.ndim != 2:
                raise TypeError("gene_prior must be a 2D numpy array of shape (n_factors, n_features)")
            if self.gene_prior.shape != (n_factors, n_features):
                raise ValueError(
                    f"gene_prior shape {self.gene_prior.shape} must match (n_factors={n_factors}, n_features={n_features})"
                )
            if np.any(self.gene_prior <= 0.0) or np.any(self.gene_prior >= 1.0):
                raise ValueError("All gene_prior probabilities must be strictly in (0, 1)")
            p_eta = self.gene_prior.astype(np.float64)
        else:
            p_eta = np.full((n_factors, n_features), 0.5, dtype=np.float64)

        log_p_eta = np.log(np.clip(p_eta, self.eps, 1.0 - self.eps))

        # 3. Resolve precision priors for t (factor precision)
        if self.alpha_t is not None:
            if not isinstance(self.alpha_t, np.ndarray) or self.alpha_t.shape != ab_t_shape:
                raise ValueError(f"alpha_t must be a 2D array of shape {ab_t_shape}")
            if np.any(self.alpha_t <= 0.0):
                raise ValueError("alpha_t values must be strictly positive (> 0)")
            alpha_t = self.alpha_t.astype(np.float64)
        else:
            alpha_t = np.ones(ab_t_shape, dtype=np.float64)

        if self.beta_t is not None:
            if not isinstance(self.beta_t, np.ndarray) or self.beta_t.shape != ab_t_shape:
                raise ValueError(f"beta_t must be a 2D array of shape {ab_t_shape}")
            if np.any(self.beta_t <= 0.0):
                raise ValueError("beta_t values must be strictly positive (> 0)")
            beta_t = self.beta_t.astype(np.float64)
        else:
            beta_t = np.ones(ab_t_shape, dtype=np.float64)

        # 4. Resolve precision priors for tau (noise precision)
        if self.alpha_tau is not None:
            if not isinstance(self.alpha_tau, np.ndarray) or self.alpha_tau.shape != ab_tau_shape:
                raise ValueError(f"alpha_tau must be a 2D array of shape {ab_tau_shape}")
            if np.any(self.alpha_tau <= 0.0):
                raise ValueError("alpha_tau values must be strictly positive (> 0)")
            alpha_tau = self.alpha_tau.astype(np.float64)
        else:
            alpha_tau = np.ones(ab_tau_shape, dtype=np.float64)

        if self.beta_tau is not None:
            if not isinstance(self.beta_tau, np.ndarray) or self.beta_tau.shape != ab_tau_shape:
                raise ValueError(f"beta_tau must be a 2D array of shape {ab_tau_shape}")
            if np.any(self.beta_tau <= 0.0):
                raise ValueError("beta_tau values must be strictly positive (> 0)")
            beta_tau = self.beta_tau.astype(np.float64)
        else:
            beta_tau = np.ones(ab_tau_shape, dtype=np.float64)

        return {
            "log_p_pi": log_p_pi,
            "log_p_eta": log_p_eta,
            "alpha_t": alpha_t,
            "beta_t": beta_t,
            "alpha_tau": alpha_tau,
            "beta_tau": beta_tau,
        }
