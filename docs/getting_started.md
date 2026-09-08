# Getting Started

> **Note**: This documentation is a **work in progress** and subject to further improvements.

WaveFactor includes a ready-to-run simulation example that creates a small spatial dataset, fits the model, and checks the results.

---

## 1. Run the Example

Make sure your environment is activated (`source wavefactor-venv/bin/activate`), then run:

```bash
python examples/getting_started/run_analysis.py
```

---

## 2. What the Example Does

The script ([`examples/getting_started/run_analysis.py`](../examples/getting_started/run_analysis.py)) runs three steps:

1. **Simulates data**: Creates a $16 \times 16$ grid (256 spots) with two clear spatial regions (left vs. right) and 100 genes.
2. **Fits WaveFactor**:
   ```python
   from wavefactor import WaveFactor

   model = WaveFactor(n_factors=2, n_length_scales=2, n_init=3, random_state=42)
   model.fit(expr, coords)
   result = model.get_result()
   ```
3. **Checks results**: Prints the model score (ELBO), verifies the spatial regions, and confirms that active genes were correctly found with high confidence ($\text{PIP} > 0.5$).
