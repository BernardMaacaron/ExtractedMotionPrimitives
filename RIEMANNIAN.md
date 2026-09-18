# Riemannian ReachGrasp Tools

The reusable Riemannian implementation lives in `riemannian/` in ExtractedMotionPrimitives.

The package uses one metric path:

```text
G(q) = M(q)
```

`M(q)` is evaluated directly from the RBDL right-arm model. Metric derivatives are estimated by finite differences of that same mass-matrix function:

```text
partial M / partial q_k ~= (M(q + eps e_k) - M(q - eps e_k)) / (2 eps)
```

## Files

- `riemannian/metric.py`: RBDL model construction, mass-matrix evaluation, finite-difference metric derivatives, and metric operations.
- `riemannian/preprocessing.py`: SciPy Savitzky-Golay smoothing and derivative estimation.
- `riemannian/segmentation.py`: Riemannian velocity-direction segmentation.
- `riemannian/geodesics.py`: geodesic integration and shooting.
- `riemannian/extraction.py`: array-to-primitive extraction.
- `scripts/extract_riemannian_primitives.py`: ReachGrasp runner using `motion_primitives.randg`.
- `notebooks/reach_grasp_geodesics.ipynb`: comparison, diagnostics, and plots using
  the shared package implementation.
- `scripts/ExtractRiemannianFull.py`: full ReachGrasp primitive extraction jobs.
