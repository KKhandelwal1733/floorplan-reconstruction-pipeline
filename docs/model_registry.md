# Model Registry

## No trained ML models are used in this pipeline.

Every stage of `roomscan` is classical geometry, signal processing, or statistics —
there is no neural network, no trained classifier, and no model weights anywhere in
this repository or its dependencies.

| Stage | Method | Not a model because |
|---|---|---|
| Gravity / floor / ceiling detection | RANSAC plane fitting (`roomscan/geometry/planes.py`) | Closed-form geometry + random sampling, no learned parameters |
| Openings (door/window) detection | Occupancy-grid gap analysis (`roomscan/geometry/openings.py`) | Rule-based grid thresholding |
| Video/photo reconstruction | Classical monocular SfM — ORB/SIFT features, essential-matrix pose recovery, DLT triangulation (`roomscan/geometry/sfm.py`) | OpenCV's standard geometric computer-vision primitives, no training |
| Scale recovery | Ensemble of hand-specified physical-size priors (`roomscan/geometry/scale_ensemble.py`) | Fixed assumptions (typical ceiling height, room diagonal, camera height), not learned |
| Damage detection | HSV colour thresholding + connected components (`roomscan/damage/detector.py`) | Hand-tuned heuristic rules, explicitly **not** a classifier (see COMPLIANCE.md) — no labelled damage dataset exists to train or validate one against |
| Confidence calibration | Split-conformal prediction (`roomscan/calibration/conformal.py`) | A statistical procedure over empirical calibration points, not a model |

## If a trained model is added later

Any future addition of a trained model (e.g. a real damage classifier, once labelled
data exists) must be registered here before merging, with: architecture, training
data provenance, validation metrics, and licence. Per the project's hard rules, this
also requires asking before adding a GPU dependency and always providing a CPU
fallback.
