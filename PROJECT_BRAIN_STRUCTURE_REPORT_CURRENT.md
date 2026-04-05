# Project Brain Structure Report - Current

Analysis date: 2026-04-05

Base paper reviewed for this update:

- "Design of an Efficient Prediction Model for Early Parkinson's Disease Diagnosis"
- IEEE Access, 2024
- DOI: 10.1109/ACCESS.2024.3421302
- Authors: K. Shyamala and T. M. Navamani

## Scope

This report documents:

- the current structure of this repository
- the active runtime and training behavior
- the relationship between the project and the base paper
- the most useful paper-driven improvements for the next phase

## 1. Executive Snapshot

This project is no longer just a paper reproduction. It is now a combined system with:

- a doctor/patient Flask portal
- runtime voice-quality screening
- multi-segment inference
- SQLite report storage
- consented feedback export for future retraining
- a research-style training pipeline with grouped validation

The current active production model is not the paper's XGBoost model. The saved runtime model is:

- dataset: `data/parkinsons_merged_aligned.data`
- source filter: `all`
- candidate: `robust_8_extra_trees`
- selected features:
  - `MDVP:Jitter(Abs)`
  - `MDVP:APQ`
  - `HNR`
  - `RPDE`
  - `DFA`
  - `spread1`
  - `spread2`
  - `PPE`
- decision threshold: `0.55`

Current saved subject-level out-of-fold metrics:

- accuracy: `0.7093`
- balanced accuracy: `0.6919`
- PD recall: `0.8163`
- healthy recall: `0.5676`
- F1 score: `0.7619`
- ROC AUC: `0.7319`
- average precision: `0.7458`

The current system is therefore:

- stronger than the original paper in deployment workflow
- weaker than the paper in paper-faithful feature ranking, explainability, and reported benchmark score

## 2. What the Base Paper Actually Does

The paper proposes an IFRX pipeline:

- preprocessing
- class balancing with `SVMSMOTE`
- feature selection using `RFE with XGBoost`
- classification comparison across 8 ML models
- explainability using `SHAP`

Paper dataset and setup:

- source: UCI Parkinson's voice dataset
- paper states 31 people total, 23 with Parkinson's
- uses the classic 22 voice biomarkers from the UCI dataset
- applies a `70/30` train-test split after preprocessing

Paper-selected 15 important features from Table 2:

1. `PPE`
2. `MDVP:Fo(Hz)`
3. `Shimmer:APQ3`
4. `D2`
5. `MDVP:Jitter(Abs)`
6. `MDVP:APQ`
7. `MDVP:Fhi(Hz)`
8. `MDVP:RAP`
9. `Shimmer:APQ5`
10. `MDVP:Jitter(%)`
11. `MDVP:Shimmer`
12. `spread2`
13. `DFA`
14. `NHR`
15. `spread1`

Paper best reported result:

- classifier: `XGBoost`
- cross-validation score: `94.24%`
- training accuracy: `100%`
- testing accuracy: `96.61%`
- precision: `97.73%`
- recall: `97.73%`
- F1 score: `97.73%`

Paper ablation message:

- `SVMSMOTE + RFE with XGBoost + SHAP + XGBoost classifier` produced the best result
- replacing or removing those parts reduced performance

Paper limitations and future work:

- training data is limited
- synthetic oversampled data may not perfectly represent real data
- wider-population validation is still needed
- multimodal data could improve performance
- other feature-selection and explainability methods should be explored
- real-time applicability should be studied further

Important interpretation note:

- The paper's score is based on a conventional split setup, not the grouped subject-level validation used in this project now.
- Because the UCI dataset contains multiple recordings per subject, I infer that the paper's reported test accuracy may be more optimistic than a strict subject-grouped evaluation.

## 3. Current Project Brain Structure

Think of this repository as a 6-part brain:

1. Interface brain
   Files: `templates/`, `static/styles.css`, `static/patient.js`, `index.html`
   Responsibility: doctor and patient flows, recording UX, waveform display, report rendering

2. Application brain
   File: `app.py`
   Responsibility: Flask routes, login/session control, database operations, upload handling, report orchestration

3. Signal brain
   File: `feature_extractor.py`
   Responsibility: quality checks, voiced-region detection, traditional biomarkers, nonlinear biomarkers, MFCC extraction

4. Dataset-engineering brain
   Files: `merge_datasets.py`, `verify_setup.py`, `data/`
   Responsibility: verifying local assets, extracting UCI-compatible features from WAV data, merging datasets, alignment statistics

5. Learning brain
   File: `train_model.py`
   Responsibility: grouped CV, candidate comparison, threshold tuning, artifact generation, evaluation export

6. Memory brain
   Files: `data/portal.db`, `models/`
   Responsibility: patient/report persistence, saved model pipeline, selected features, evaluation history

## 4. Current End-to-End System Flow

### Runtime flow

```text
Doctor or patient
    ->
Flask routes in app.py
    ->
Upload or browser recording
    ->
Temporary WAV
    ->
Split into overlapping voice segments
    ->
feature_extractor.extract_features()
    ->
prediction_pipeline.pkl + selected features + threshold
    ->
segment aggregation + quality assessment
    ->
SQLite report save
    ->
doctor/patient dashboard rendering
```

### Training flow

```text
UCI data + optional merged/aligned WAV-derived rows
    ->
subject ID derivation
    ->
candidate subset/model selection
    ->
nested StratifiedGroupKFold validation
    ->
SMOTE inside training folds
    ->
subject-level threshold tuning
    ->
best model save
    ->
evaluation CSV / JSON / SVG artifacts
```

## 5. Current Data and Model State

### Training datasets currently available

- `data/parkinsons.data`
  - original UCI dataset
  - 195 rows
  - 22 numeric voice features

- `data/parkinsons_merged.data`
  - raw merged UCI + extracted WAV rows

- `data/parkinsons_merged_aligned.data`
  - aligned merged variant used by the current active model

### Current merged-data snapshot

From the latest verified merge:

- original UCI rows: `195`
- added PD WAV rows: `25`
- added healthy WAV rows: `29`
- total merged rows: `249`
- skipped WAV files: `27`

Main skipped-reason counts:

- `14` too short
- `10` insufficient stable pitch detail for nonlinear extraction
- `3` clipped/distorted

### Operational database snapshot

At the time of this update:

- total users: `2`
- doctors: `1`
- patients: `1`
- reports: `2`

### Active runtime model

The runtime model is loaded from:

- `models/prediction_pipeline.pkl`
- `models/selected_features.pkl`
- `models/model_metadata.json`

Important active runtime settings:

- selected feature subset: `robust_8`
- threshold: `0.55`
- gender-specific extraction flag: `false`

That last point is intentional:

- training WAV extraction was done without gender-specific pitch ranges
- runtime now matches that behavior to avoid train/inference mismatch

## 6. Paper-to-Project Alignment

### Where the project aligns well with the paper

- It still uses the UCI Parkinson's voice feature family as the training backbone.
- It uses standardization in the training pipeline.
- It explicitly handles class imbalance.
- It experiments with multiple classifiers and feature subsets.
- It keeps `PPE`, `DFA`, `RPDE`, `MDVP:Jitter(Abs)`, and `MDVP:APQ`, which are all paper-relevant features.

### Where the project intentionally diverges from the paper

- The paper uses `SVMSMOTE`; the project currently uses `SMOTE`.
- The paper's core classifier is `XGBoost`; the current active model is `ExtraTreesClassifier`.
- The paper uses `RFE with XGBoost` to rank 15 features; the current project uses manually curated subsets.
- The paper uses `SHAP`; the current project does not yet generate explainability outputs.
- The paper uses a `70/30` split; the project uses nested grouped CV with subject-level aggregation.
- The project includes real portal workflows, WAV ingestion, quality checks, segment aggregation, and retraining export, which are outside the paper's scope.

### Important overlap between paper features and current active model

The current `robust_8` subset overlaps strongly with the paper's selected features:

- `PPE`
- `MDVP:Jitter(Abs)`
- `MDVP:APQ`
- `DFA`
- `spread1`
- `spread2`

Features from the paper's top-15 list that are not in the current active subset:

- `MDVP:Fo(Hz)`
- `Shimmer:APQ3`
- `D2`
- `MDVP:Fhi(Hz)`
- `MDVP:RAP`
- `Shimmer:APQ5`
- `MDVP:Jitter(%)`
- `MDVP:Shimmer`
- `NHR`

## 7. Current Research Findings Inside This Project

During the recent project benchmarking work:

- `aligned + all sources + robust_8_extra_trees` performed better than raw merged data
- `figshare-only` training performed worse than `all sources`
- threshold `0.55` was already the best practical operating point among the tested range
- the biggest weakness remains healthy recall, not PD recall

Current research takeaway:

- The project should not abandon UCI data yet.
- The next serious research step is not threshold tuning.
- The next serious research step is to add a paper-faithful `XGBoost + RFE + explainability` benchmark and compare it against the current grouped-CV production baseline.

## 8. What the Paper Suggests We Should Build Next

The paper is useful, but it should guide a research branch, not blindly replace the current production model.

### High-priority next improvements

1. Add a paper-faithful benchmark to `train_model.py`
   - classifier: `XGBoost`
   - feature selection: `RFE with XGBoost`
   - target feature count: `15`
   - explainability export: `SHAP`

2. Evaluate the paper-faithful benchmark under two conditions
   - paper-like reproduction on `UCI-only`
   - production-safe grouped subject-level evaluation on the current aligned dataset

3. Add an ablation report inside the repo
   Compare:
   - current `robust_8_extra_trees`
   - `paper_15_xgboost`
   - SMOTE vs SVMSMOTE
   - UCI-only vs aligned merged

4. Export explainability artifacts for doctor review
   - global feature importance
   - per-prediction explanation for saved reports

### Medium-priority improvements inspired by the paper

1. Add `XGBoost` as a first-class training candidate
2. Add `SVMSMOTE` as an optional resampler in the training script
3. Compare the current curated subsets against the paper's ranked 15-feature subset
4. Add a model-comparison markdown or JSON summary after each training run

### Improvements the paper supports but should be delayed

1. Multimodal learning with demographics or clinical data
2. More aggressive synthetic-data generation
3. Replacing the production model purely based on paper-style accuracy

## 9. Practical Recommendation for This Project

Use the paper as the research foundation, but keep the current project standards for deployment:

- Keep grouped subject-level validation as the main truth for model selection.
- Keep the current portal workflow and quality gate architecture.
- Do not switch the runtime model to a paper-style XGBoost model unless it beats the current grouped-CV baseline.
- Add SHAP because the paper is right that interpretability matters for healthcare-facing tools.
- Add a paper-faithful benchmark because right now the project references the paper conceptually, but does not reproduce its actual model design.

## 10. Immediate Action Order

If we continue from here, the most logical order is:

1. Install `xgboost` and `shap` in the runtime environment
2. Run the newly added `paper_15_extra_trees` and `paper_15_xgboost` benchmarks
3. Compare `SMOTE` vs `SVMSMOTE` fairly under grouped subject-level validation
4. Export SHAP explanations into `models/evaluation/`
5. Compare the new paper-faithful branch against the current `robust_8_extra_trees` model
6. Only then decide whether to replace the active runtime model

### Latest implementation updates

- Upload analysis now accepts common audio formats, not only WAV.
- Non-WAV uploads are normalized to temporary WAV on the server before segmentation.
- Short-but-decodable recordings are no longer hard-failed just because they produce fewer than three valid segments; they can still be analyzed and marked as `retake`.
- The training script now includes the paper-ranked `paper_15` feature subset and supports `--resampler svmsmote`.
- A `paper_15_xgboost` candidate is wired in and becomes available automatically once `xgboost` is installed.

## 11. Architectural Conclusion

The current repository is best understood as:

```text
Paper-inspired research core
    +
production-style voice portal
    +
subject-grouped training and evaluation pipeline
```

The paper gave this project its starting point.
The current codebase has already moved beyond a direct paper copy.
The right next step is not to throw away the current system, but to add a clean paper-faithful benchmark path and compare it fairly against the stronger deployment-oriented pipeline already present in this repository.
