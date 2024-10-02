# WaviFM: Wavelet-based variationally-inferred Factor Model

Welcome!

This repository contains the implementation for the method WaviFM, a Wavelet-based variationally-inferred Factor Model. It also contains example analysis code that reproduces the figures and results from the thesis concerning WaviFM.

## Guide to using WaviFM

Note: environmental setup described below only describe one setting which works, in practice many of the requirement can be relaxed (especially less relevant ones to the code and compilation, such as the VSCode version, Windows version)

### Reproducing results from the thesis for WaviFM

Environmental setup

- Python 3.7.12
  - Regarding required packages, I recommend creating a Python environment with the provided environment file `./environment.yml`. For example, this can be done in Anaconda prompt by placing this file in the current directory of Anaconda prompt and running `conda env create -f environment.yml`. The resulting environment is called `py3712`
    - Doing so will likely install some redundant packages, but at least will not be missing package.
  - However, if one wishes to manually install packages, below lists the main packages that should be installed (alongside their dependencies):
    - pybind11
    - ipykernel
    - ipython
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - pywt
    - scipy
    - scikit-learn
- R 4.4.1
  - Required packages (and their dependencies) include:
    - BiocManager
    - STexampleData
    - devtools
    - OliverXUZY/waveST (on Github)
  - Above packages can be installed by running the following code in R

```
# Install dependency
install.packages("tidyverse")
install.packages("fields")

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("STexampleData")
install.packages("devtools")

library(devtools)
install_github("OliverXUZY/waveST")
```

Below gives step by step instructions to reproducing results from individual sections in the thesis. This was tested to work with the provided compiled Python bindings file `./build/WaviFM.cp37-win_amd64.pyd` is present in the repository. Compiling the bindings yourself should also work.

Reproducing section 4.1.3: 

1. Run every Jupyter notebook in `./examples/wavelet_reconstruction/`. In this directory, the Jupyter notebook `wavelet_reconstruction_2M.ipynb` and `wavelet_reconstruction_10M.ipynb` reproduces results from the thesis for the data set simulated from 2 and 10 ground truth factors, respectively.

Reproducing section 4.1.4:

1. Run every Jupyter notebook in `./examples/use_spatial_info/`. In this directory, the Jupyter notebook `polygons.ipynb`, `polygonsPermuted.ipynb`, `rectangles.ipynb`, `rectanglesPermuted.ipynb` reproduces results from the thesis for WaviFM and PCA applied on their namesake data sets. Note that PCA was still ran for spot-permuted data sets so that comparison with PCA inference from non-spot-permuted data sets enable confirmation that indeed spot permutation does not affect PCA inference (up to minor numerical differences).

Reproducing section 4.1.5:

1. Run every Jupyter notebook in `./examples/multilength_robustness/`. In this directory, each Jupyter notebook reproduces the RRMSE values for WaviFM, Smoothed PCA, PCA applied to their namesake data sets in the thesis.

Reproducing section 4.1.6: note that exported results for steps 1, 2 are already included in the repository, so one can skip them if one only wants to reproduce the figures without running inference and other preceding steps again.

1. Run `./examples/wavest_comparison/data_simulation_and_wavest_inference.Rmd`. This simulates the data, exports the data, and runs waveST inference on the data.
2. Run `./examples/wavest_comparison/wavifm_inference.ipynb`. This runs WaviFM inference on the data exported in the previous step and exports the inference result.
3. Run `./examples/wavest_comparison/comparison_analysis.ipynb`. This generates the figures for this section and compares the WaviFM and waveST inference results. 

Reproducing section 4.2.1: note that exported results for steps 1, 2, 3 are already included in the repository, so one can skip them if one only wants to reproduce the figures without running inference and other preceding steps again.

1. Run `./examples/human_dlpfc/dlpfc_preprocessing_and_to_csv.ipynb`. This loads the DLPFC data, preprocesses this data, and exports it to CSV format use in ensuing inference.
2. Run `./examples/human_dlpfc/dlpfc_wavest_inference.Rmd`. This runs waveST inference on the DLPFC data.
3. Run `./examples/human_dlpfc/dlpfc_wavifm_pca_inference.ipynb`. This runs WaviFM and PCA inference on the DLPFC data.
4. Run `./examples/human_dlpfc/dlpfc_analysis.ipynb`.  This generates the figures for this section and compares the WaviFM, PCA and waveST inference results.

Reproducing Appendix C:

1. Run `./examples/wavelet_inference/wavelet_inference.ipynb`.

### Optional: compiling Python bindings for the CAVI part of WaviFM from C++ source

Environmental setup

- Windows 10 Pro, Version 10.0.19045 Build 19045 (I have not verified on whether compilation works on other operating systems)
- VSCode Version 1.91.0, with the following extensions
  - C/C++
  - C/C++ Extension Pack
  - CMake
  - CMake Tools
- CMake Version 3.30
- G++ 13.2.0 (Rev6, Built by MSYS2 project)
- Python 3.7.12
  - Regarding required packages, I recommend creating a Python environment with the provided environment file `./python_environment.yml`. For example, this can be done in Anaconda prompt by placing this file in the current directory of Anaconda prompt and running `conda env create -f python_environment.yml`. The resulting environment is called `py3712`
    - Doing so will likely install some redundant packages, but at least will not be missing package.
  - However, if one wishes to manually install packages, below lists the main packages that should be installed (alongside their dependencies):
    - pybind11

Depending on your environment, there may be additional setups required to get C++ and CMake running with VSCode. The following linked series of tutorials for C++ and CMake integration in VSCode may be of use (https://code.visualstudio.com/docs/cpp/introvideos-cpp).

Steps to compiling Python bindings for CAVI

1. Open this repository with VSCode
2. Delete the `./build` directory. This helps avoid any conflicts with your compilation and existing compilation results.
3. Edit `set(PYBIND11_PYTHON_VERSION 3.7.12)` and `set(PYTHON_EXECUTABLE "C:/Users/yw/miniconda3/envs/py3712/python.exe")` in `./CMakeLists.txt` respectively to match your Python version and the corresponding Python executable path.
4. Run `CMake:Build` command in VSCode
5. The Python bindings should appear in `.\build\` with suffix `.pyd` and a name that depends on the Python version and environment used for compilation. An example of the full filename of such generated binding is `WaviFM.cp37-win_amd64.pyd`.

You may also wish to run the tests for the C++ code. This can be done by running the `Cmake:Run Tests` command in VSCode

## Repository overview

`.\build`: contains CMake generated files, albeit currently only contains `WaviFM.cp37-win_amd64.pyd` which is the Python 3.7 bindings (compiled from C++ source code) for the CAVI part of WaviFM. This allows one to use the CAVI part of WaviFM and in turn WaviFM as a whole without needing to compile the C++ source code.

`.\data`: contains DLPFC data

`.\eigen`: contains eigen library for C++, used for some numerical computations in C++ source code.

`.\examples`: contains examples of applications of WaviFM that reproduces the figures and results from the thesis concerning WaviFM.

`.\googletest`: contains googletest library for C++, used for automated testing of C++ source code.

`.\pybind11`: contains pybind11 library for C++, used for compiling Python bindings (from C++ source code) for the CAVI part of WaviFM.

`.\scripts`: contains python scripts with containing various functionalities, including running WaviFM, plotting, etc.

`.\src`: contains C++ source code for the CAVI part of WaviFM

`.\test`: contains testing code for the C++ source code for the CAVI part of WaviFM