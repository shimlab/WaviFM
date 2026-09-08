# WaveFactor: Bayesian Multiresolution Wavelet Spatial Factor Model

**WaveFactor** (formerly `WaviFM`) is a Bayesian factor modeling framework for spatial transcriptomics that explicitly models spatial length scales by performing Coordinate Ascent Variational Inference (CAVI) directly on 2D Discrete Wavelet Transform (DWT) coefficients.

> **Documentation Status**: Documentation is currently a **work in progress** and subject to further improvements. See the [docs/](docs/README.md) directory for guides and specifications.

---

## 🚀 Installation

WaveFactor requires Python $\ge 3.7$, CMake $\ge 3.16$, and a C++17 compiler (e.g. GCC or Clang).

To install WaveFactor in an isolated virtual environment:

```bash
# 1. Clone this repository (copy URL from the green "Code" button on GitHub)
git clone <repository-url>
cd <repository-folder>

# 2. Create and activate an isolated virtual environment
python3 -m venv wavefactor-venv
source wavefactor-venv/bin/activate

# 3. Install WaveFactor
pip install .
```

For further details, see the **[Installation Guide](docs/installation.md)**.

---

## 🔬 Getting Started Example

WaveFactor includes a complete, self-contained getting started script that simulates a $16 \times 16$ lattice (256 spots, 2 spatial domains, 100 genes), fits the model, and validates recovery against ground truth:

```bash
python examples/getting_started/run_analysis.py
```

See **[Getting Started Guide](docs/getting_started.md)** for more details.

---

## 📊 Quick API Example

```python
import wavefactor as wf

# 1. Instantiate WaveFactor estimator
model = wf.WaveFactor(
    n_factors=10,               # Number of latent factors (K)
    n_length_scales=4,          # Wavelet detail levels (R = 5 total resolutions)
    n_init=5,                   # Multi-start initializations (selects best ELBO)
    n_jobs=-1,                  # Parallel workers across initializations
    random_state=42,
)

# 2. Fit and transform spatial transcriptomics data
# Input requirements:
#   X: continuous expression matrix (N_spots x N_genes)
#   coords: (N_spots x 2) integer grid coordinates covering [0, L-1] x [0, L-1]
#   N_spots = L * L must be an exact power of 4 (e.g. 64, 256, 1024, 4096)
factors = model.fit_transform(X, coords)

# 3. Access rich posterior estimates
result = model.get_result()
print(f"Final ELBO: {result.elbo:.2f} across {result.n_iter} iterations")

spot_factors = result.factors        # Latent spatial factors (N_spots x K)
gene_loadings = result.loadings      # Factor loadings matrix (K x N_genes)
gene_pips = result.gene_pip          # Gene Posterior Inclusion Probabilities
spatial_pips = result.spatial_pip    # Multiresolution spatial wavelet PIPs
```

For detailed input constraints (lattice power-of-2 side lengths, preprocessing considerations) and outputs, see **[Data Input & Output Specifications](docs/input_output.md)**.

---

## 🧪 Testing & Quality Assurance

Run both the compiled C++17 GoogleTests (50 tests) and Python test suite with a single command:

```bash
# Run ALL tests (automatically compiles/checks C++ targets)
python tests/run_all_tests.py
```

Options:
- `python tests/run_all_tests.py --py-only` : Run only Python unit & parity tests.
- `python tests/run_all_tests.py --cpp-only`: Run only C++ GoogleTests.
- `python tests/run_all_tests.py -v`        : Verbose test output.

---

## 📂 Repository Structure

- `docs/`: Standalone markdown documentation ([index](docs/README.md), [installation](docs/installation.md), [input/output](docs/input_output.md), [getting started](docs/getting_started.md)).
- `examples/`: Example scripts and case studies ([getting started](examples/getting_started/run_analysis.py)).
- `wavefactor/`: Scikit-Learn compatible Python package (`WaveFactor`, `WaveFactorResult`, `WaveFactorData`, `Priors`).
- `src/`: C++17 CAVI engine source files (`cavi.cpp`, `updates.cpp`, `elbo.cpp`, `parameters.cpp`, `bindings.cpp`).
- `test/`: GoogleTest C++ unit testing suite (`cavi_test.cpp`, `updates_test.cpp`, `elbo_test.cpp`, `tensor_test.cpp`, `utilities_test.cpp`).
- `tests/`: Python test suite (`test_priors.py`, `test_data.py`, `test_model.py`, `test_reference_parity.py`, `run_all_tests.py`).
- `tests/fixtures/`: Ground-truth reference datasets (`reference_input.csv`, `reference_output.npz`) for bitwise parity auditing.