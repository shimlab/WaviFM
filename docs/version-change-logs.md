This folder contains logs regarding changes and/or notes made to software made in particular versions. This file doesn't necessarily include all version logs.

## v1.0.0

Notes
- Version used in Master's Thesis

## v1.1.0

Changes
- Modified model to allow free prior setting for $\pi_{F,m,g}$ describing sparsity prior on a per factor m and per gene g basis (formerly only allowed setting $\pi_{F,m}$ which specifies an overall sparsity for factor m across all genes)
- Various other changes

## v1.2.0

Changes
- Added and tested for Linux platform compatibility (formerly only tested on Windows platform)
- Various other changes

## v1.3.0

Changes
- Added hard coded edge case for $\pi_{F,m,g}$ values being set as 0 and/or 1 (within a small interval since floating point equivalence comparison doesn't work exactly for single values)
- Various other changes

## v1.3.1

Changes
- Updated prebuilt binary for code