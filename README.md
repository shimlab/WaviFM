# WaveFactor: Bayesian Multiresolution Wavelet Spatial Factor Model

**WaveFactor** (formerly `WaviFM`) is a Bayesian factor modeling framework for spatial transcriptomics that explicitly models spatial length scales by performing Coordinate Ascent Variational Inference (CAVI) directly on 2D Discrete Wavelet Transform (DWT) coefficients.

---

## 🚀 Quick Start (WaveFactor v2.0)

### Installation

Install the package and compile the C++ extension module using `pip`:

```bash
# Clone the repository
git clone https://github.com/shimlab/WaviFM.git
cd WaviFM

# Install in editable mode
pip install -e .
```

### Modern Python API (Scikit-Learn Style)

WaveFactor v2.0 provides an ergonomic Scikit-Learn compatible estimator that automatically handles spatial gridding, feature standardization, 2D DWT, and automated 2D Inverse DWT (IDWT) back to spot space:

```python
import numpy as np
import wavefactor as wf

# 1. Instantiate WaveFactor model
# Priors default to standard canonical baselines if omitted, or can be passed as exact arrays:
#   spatial_prior: 1D array of shape (R,) in (0, 1)
#   gene_prior: 2D array of shape (K, G) in (0, 1)
#   alpha_t, beta_t: 2D arrays of shape (R, K) > 0
#   alpha_tau, beta_tau: 2D arrays of shape (R, G) > 0
model = wf.WaveFactor(
    n_factors=10,               # Number of latent factors (K)
    n_length_scales=4,          # Wavelet detail levels (R = 5 resolutions)
    n_init=5,                   # Multi-start initializations (picks best ELBO)
    n_jobs=-1,                  # Parallel workers across initializations
    random_state=42,
)

# 2. Fit and transform spatial transcriptomics data
# X: expression matrix (N_spots x N_genes) where N_spots is an exact power of 4 (e.g. 64, 256, 1024, 4096)
# coords: (N_spots x 2) integer lattice coordinates covering the [0, L-1] x [0, L-1] grid (L = sqrt(N_spots))
factors = model.fit_transform(X, coords)

# 3. Access rich posterior estimates
result = model.get_result()
print(f"Final ELBO: {result.elbo:.2f} across {result.n_iter} iterations")

# Spatial factors in spot space (N_spots x K)
spot_factors = result.factors

# Gene loadings matrix (K x N_genes)
gene_loadings = result.loadings

# Posterior inclusion probabilities (PIPs)
gene_pip = result.gene_pip          # (K x N_genes)
spatial_pip = result.spatial_pip    # Multiresolution wavelet PIPs

```

---

## 🧪 Testing & Quality Assurance

WaveFactor includes a unified test runner that executes **both** the compiled C++17 GoogleTest mathematical suite (50 tests) and the Python unit/integration test suite with a single command:

```bash
# Run ALL tests (automatically verifies and recompiles C++ targets if modified)
python tests/run_all_tests.py
```

### Selective Test Execution

```bash
# Run only Python tests
python tests/run_all_tests.py --py-only

# Run only C++ GoogleTests
python tests/run_all_tests.py --cpp-only

# Skip incremental CMake build check
python tests/run_all_tests.py --no-build

# Verbose output
python tests/run_all_tests.py -v
```

### Direct Tooling Invocations

- **C++ GoogleTests via CMake / CTest**:
  ```bash
  cmake --build build --target WaviFMTests
  ctest --test-dir build --output-on-failure
  ```
- **Python Tests via Pytest**:
  ```bash
  pytest tests/ -v
  ```

---

## 📂 Repository Structure

- `wavefactor/`: Modern Python package (`WaveFactor`, `WaveFactorResult`, `WaveFactorData`, `Priors`).
- `src/`: C++17 CAVI engine source files (`cavi.cpp`, `updates.cpp`, `elbo.cpp`, `parameters.cpp`, `bindings.cpp`).
- `test/`: GoogleTest C++ unit testing suite (`cavi_test.cpp`, `updates_test.cpp`, `elbo_test.cpp`, `tensor_test.cpp`, `utilities_test.cpp`).
- `tests/`: Python test suite (`test_priors.py`, `test_data.py`, `test_model.py`, `test_reference_parity.py`, `run_all_tests.py`).
- `tests/fixtures/`: Ground-truth reference datasets (`reference_input.csv`, `reference_output.npz`) for bitwise parity auditing.