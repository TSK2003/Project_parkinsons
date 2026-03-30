# Project Brain Structure Report

Analysis date: 2026-03-30

## Scope

This report analyzes the project source, templates, static assets, local data files, model artifacts, and evaluation outputs inside this repository.

Excluded from detailed line-by-line analysis:

- `.git/` because it is version-control metadata, not application logic
- `venv/` because it is an environment folder, not project source code

## 1. Brain Structure View

Think of this project as a 5-part brain:

1. Presentation brain
   Files: `templates/`, `static/styles.css`, `static/patient.js`, `index.html`
   Responsibility: doctor and patient UI, recording workflow, waveform display, status messages

2. Application brain
   File: `app.py`
   Responsibility: Flask routes, session/auth handling, SQLite access, report creation, audio segmentation, model inference orchestration

3. Signal-processing brain
   File: `feature_extractor.py`
   Responsibility: audio quality checks, voiced-region isolation, perturbation features, nonlinear voice biomarkers

4. Learning brain
   File: `train_model.py`
   Responsibility: dataset loading, grouped cross-validation, feature-subset comparison, threshold tuning, artifact generation

5. Memory brain
   Files: `data/portal.db`, `data/parkinsons.data`, `models/`
   Responsibility: patient/report persistence, training dataset storage, saved model/evaluation artifacts

## 2. End-to-End System Flow

### Runtime flow

```text
Doctor or Patient Browser
    ->
Flask routes in app.py
    ->
Audio upload or browser recording
    ->
Temporary WAV file
    ->
Audio segmentation into overlapping windows
    ->
feature_extractor.extract_features()
    ->
prediction_pipeline.pkl or legacy fallback model
    ->
quality assessment + summary generation
    ->
SQLite report save in data/portal.db
    ->
Rendered dashboards and report cards
```

### Training flow

```text
data/parkinsons.data
    ->
subject ID extraction from recording names
    ->
nested StratifiedGroupKFold validation
    ->
candidate feature subsets + candidate classifiers
    ->
threshold tuning at subject level
    ->
best final model selection
    ->
saved artifacts in models/
```

## 3. Datasets and Stored Data

### Dataset 1: `data/parkinsons.data`

- Purpose: training dataset for the voice-based Parkinson's classifier
- Shape: 195 rows x 24 columns
- Columns: 1 recording identifier column (`name`), 1 label column (`status`), 22 numeric voice features
- Missing values found: 0
- Status distribution:
  - `0` (healthy): 48 rows
  - `1` (Parkinson's): 147 rows
- Subject count derived from recording prefixes: 32

Feature columns used in the dataset:

```text
MDVP:Fo(Hz)
MDVP:Fhi(Hz)
MDVP:Flo(Hz)
MDVP:Jitter(%)
MDVP:Jitter(Abs)
MDVP:RAP
MDVP:PPQ
Jitter:DDP
MDVP:Shimmer
MDVP:Shimmer(dB)
Shimmer:APQ3
Shimmer:APQ5
MDVP:APQ
Shimmer:DDA
NHR
HNR
RPDE
DFA
spread1
spread2
D2
PPE
```

### Dataset 2: `data/portal.db`

- Purpose: operational database for doctor accounts, patient accounts, and saved reports
- Storage engine: SQLite
- Tables present:
  - `users`
  - `reports`
  - `sqlite_sequence`
- Current snapshot at analysis time:
  - total users: 2
  - doctors: 1
  - patients: 1
  - reports: 1

### Model storage: `models/`

- Purpose: serialized model pipeline, selected feature list, legacy feature-selection artifacts, and evaluation outputs
- Important behavior:
  - runtime prefers `prediction_pipeline.pkl`
  - runtime also reads `model_metadata.json`
  - legacy fallback uses `parkinsons_model.pkl`, `scaler.pkl`, and optionally `rfe_selector.pkl`

## 4. Main Modules and Dependencies

### Python package dependencies from `requirements.txt`

- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `flask`
- `flask-cors`
- `joblib`
- `praat-parselmouth`
- `nolds`
- `scipy`

### Standard-library modules used heavily

- `json`
- `os`
- `sqlite3`
- `tempfile`
- `datetime`
- `functools`

### Browser APIs used by the frontend

- `navigator.mediaDevices.getUserMedia`
- `MediaRecorder`
- `AudioContext` / `webkitAudioContext`
- `CanvasRenderingContext2D`
- `fetch`
- `FormData`

## 5. File-by-File Analysis

## Top-Level Files

### `.gitignore`

- Purpose: excludes Python cache files, environment folders, coverage artifacts, and project-specific runtime artifacts
- Important detail: both `data/` and `models/` are ignored
- Meaning: the dataset, SQLite database, and trained model artifacts are intended to be local/generated, not committed as normal source files

### `README.md`

- Purpose: project overview and quick-start guide
- Documents:
  - Flask portal usage
  - doctor default login
  - patient flow
  - doctor flow
  - API endpoints
  - retraining command
- Important note: README still says "patient registration and login", but the current code disables self-registration and redirects `/register` to patient login

### `requirements.txt`

- Purpose: Python dependency list
- Note: package versions are not pinned
- Risk: unpinned ML dependencies can cause model pickle compatibility issues across environments

### `index.html`

- Purpose: standalone static landing page that tells the user to start `app.py`
- It is not part of the Flask template inheritance chain
- Role: helper/launcher page, not the main application UI

### `app.py`

- Purpose: central Flask application and runtime orchestration layer
- Core responsibilities:
  - initialize app config and CORS
  - load model pipeline or legacy artifacts
  - create and migrate SQLite schema
  - seed default doctor account
  - manage sessions and role-based access
  - analyze uploaded/recorded audio
  - save reports and render dashboards

Key configuration/constants:

- data/model paths:
  - `DATA_DIR`
  - `MODELS_DIR`
  - `DATABASE_PATH`
  - `MODEL_PIPELINE_PATH`
  - `MODEL_METADATA_PATH`
  - `LEGACY_MODEL_PATH`
  - `LEGACY_SCALER_PATH`
  - `LEGACY_RFE_PATH`
- feature list:
  - `ALL_FEATURES` defines the 22 expected biomarkers
- runtime quality settings from env or defaults:
  - `SEGMENT_SECONDS = 3.0`
  - `SEGMENT_OVERLAP_SECONDS = 1.0`
  - `MIN_VALID_SEGMENTS = 2`
  - `MAX_SKIPPED_SEGMENT_RATIO = 0.34`
  - `MIN_SEGMENT_STABILITY = 0.70`
  - `MIN_CONFIDENCE_GAP = 0.18`

Database responsibilities inside `app.py`:

- `init_db()`
  - creates `users` and `reports` tables if missing
- `ensure_user_columns()`
  - adds `mobile_number` if older DB schema is missing it
- `ensure_report_columns()`
  - adds `quality_status`, `needs_retake`, `quality_flags_json` if missing
- `ensure_default_doctor()`
  - seeds doctor account from env or defaults to `doctor` / `doctor123`

Main user/account logic:

- `create_patient_record()`
  - validates name/mobile/password
  - generates patient ID from first 4 alphanumeric name chars plus last 4 mobile digits
  - stores hashed password
- `delete_patient_record()`
  - removes patient and cascades report deletion
- `process_login()`
  - shared doctor/patient login logic
- `login_required()`
  - role-aware route guard

Report storage logic:

- `save_report()`
  - writes scalar predictions plus JSON blobs for features, segment results, skipped segments, and quality flags
- `hydrate_report()`
  - converts JSON strings back into Python objects for template/API use

Inference logic:

- `infer_prediction()`
  - active path: `prediction_pipeline.pkl` + `selected_model_features` + tuned threshold from metadata
  - fallback path: scaler -> optional RFE selector -> legacy classifier
- `run_prediction()`
  - extracts features from one audio file and returns label/probabilities/features
- `split_audio_segments()`
  - breaks audio into overlapping windows for multi-segment aggregation
- `assess_report_quality()`
  - evaluates segment count, skipped ratio, stability, and confidence gap
- `build_report_summary()`
  - generates human-readable report summary
- `analyze_voice_sample()`
  - full multi-window workflow with averaging, stability, retake logic, and cleanup

HTTP routes:

- `GET /`
  - home page or dashboard redirect if logged in
- `GET /health`
  - health response JSON
- `GET/POST /register`
  - no longer registers users; always redirects to patient login with flash message
- `GET/POST /login/patient`
  - patient login
- `GET/POST /login/doctor`
  - doctor login
- `GET /logout`
  - session clear
- `GET /patient/dashboard`
  - patient report dashboard
- `POST /doctor/patients/create`
  - doctor creates patient account
- `GET /doctor/dashboard`
  - doctor overview
- `GET /doctor/patient/<id>`
  - doctor patient detail page
- `POST /doctor/patient/<id>/delete`
  - doctor deletes patient
- `POST /doctor/patient/<id>/reports`
  - doctor uploads/records and saves report for patient
- `POST /api/reports`
  - patient uploads/records and saves own report
- `POST /stream-chunk`
  - legacy single-chunk prediction endpoint
- `POST /analyze-voice`
  - analyze without saving

Important design characteristics:

- application startup runs `init_db()` inside `app.app_context()`
- uses temporary files for uploaded audio
- uses `secure_filename()` before storing source filename
- stores report details in normalized scalar columns plus JSON columns

### `feature_extractor.py`

- Purpose: voice biomarker extraction from audio
- Role: this is the scientific/signal-processing engine of the app

Quality gate constants:

- minimum duration: `2.0` seconds
- minimum RMS: `0.002`
- minimum voiced seconds: `0.7`
- minimum voiced frames: `20`
- max clipped fraction: `0.01`
- target sample rate: `44100`
- pitch floor/ceiling: `75` to `600`

Pipeline inside `extract_features()`:

1. Load audio with Parselmouth
2. Convert stereo to mono if needed
3. Reject too-short, too-quiet, or clipped audio
4. Normalize and resample to 44.1 kHz
5. Find the longest stable voiced region
6. Use the first half of the voiced region for traditional perturbation metrics when long enough
7. Extract traditional features
8. Extract nonlinear features
9. Return merged feature dictionary

Traditional features extracted:

- `MDVP:Fo(Hz)`
- `MDVP:Fhi(Hz)`
- `MDVP:Flo(Hz)`
- `MDVP:Jitter(%)`
- `MDVP:Jitter(Abs)`
- `MDVP:RAP`
- `MDVP:PPQ`
- `Jitter:DDP`
- `MDVP:Shimmer`
- `MDVP:Shimmer(dB)`
- `Shimmer:APQ3`
- `Shimmer:APQ5`
- `MDVP:APQ`
- `Shimmer:DDA`
- `NHR`
- `HNR`

Nonlinear features extracted:

- `RPDE`
- `DFA`
- `spread1`
- `spread2`
- `D2`
- `PPE`

Important helper responsibilities:

- `_normalize_and_resample()`
  - centers waveform and rescales peak to 0.95
- `_extract_longest_voiced_region()`
  - finds stable voiced pitch frames and isolates the best region
- `_extract_traditional_features()`
  - uses Praat point-process and harmonicity measures
- `_extract_nonlinear_features()`
  - uses downsampled waveform and pitch-series analysis
- `_pitch_distribution_features()`
  - derives `spread1`, `spread2`, `PPE`
- `_rpde()`
  - custom recurrence period density entropy implementation

Design observation:

- the code is intentionally aligned to the training dataset assumptions by normalizing amplitude and resampling to 44.1 kHz before feature computation

### `train_model.py`

- Purpose: full model-training and evaluation pipeline
- Role: converts `data/parkinsons.data` into serialized runtime artifacts under `models/`

Global training settings:

- random state: `42`
- outer grouped CV splits: `4`
- inner grouped CV splits: `4`
- target PD recall: `0.85`
- target balanced accuracy: `0.75`
- threshold grid: `0.25` to `0.85` in 25 steps

Feature subsets defined:

- `paper_4`
  - `HNR`, `RPDE`, `DFA`, `PPE`
- `paper_6`
  - `MDVP:Jitter(Abs)`, `Jitter:DDP`, `MDVP:APQ`, `HNR`, `RPDE`, `DFA`, `PPE`
- `nonlinear_5`
  - `HNR`, `RPDE`, `DFA`, `D2`, `PPE`
- `robust_8`
  - `MDVP:Jitter(Abs)`, `MDVP:APQ`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `PPE`

Important note:

- `paper_6` is named as if it has 6 features, but the list currently contains 7 features
- `robust_8` is defined but not actually used in `build_candidate_configs()`

Candidate models actually evaluated:

- `nonlinear_5_extra_trees`
- `paper_4_extra_trees`
- `paper_4_logreg`
- `paper_6_random_forest`

Core functions:

- `extract_subject_ids()`
  - groups repeated recordings into subject-level IDs such as `phon_R01_S01`
- `aggregate_subject_predictions()`
  - averages recording probabilities into subject predictions
- `compute_subject_metrics()`
  - balanced accuracy, sensitivity, specificity, F1, ROC AUC, average precision
- `tune_threshold()`
  - chooses best decision threshold over subject probabilities
- `evaluate_candidate_with_inner_cv()`
  - grouped inner-CV evaluation and threshold selection
- `write_curve_csv()`, `write_line_chart_svg()`, `write_confusion_matrix_artifacts()`
  - save evaluation outputs
- `main()`
  - orchestrates full training run

Training stages implemented in `main()`:

1. Load dataset and derive subject groups
2. Run nested grouped validation with threshold tuning
3. Aggregate out-of-fold predictions
4. Write evaluation artifacts
5. Choose final model and threshold on all subjects
6. Save inference artifacts
7. Print output locations

Artifacts written by training:

- `prediction_pipeline.pkl`
- `parkinsons_model.pkl`
- `scaler.pkl`
- `selected_features.pkl`
- `model_metadata.json`
- evaluation CSV/JSON/SVG files

Notably not written by current training code:

- `rfe_selector.pkl`
- `selected_mask.pkl`

That means those two files are legacy leftovers from an older training workflow.

## Template Files

### `templates/base.html`

- Purpose: shared layout wrapper
- Provides:
  - page shell
  - branded header
  - current user display
  - login/logout nav
  - flashed message stack
  - overridable blocks for title, head, content, and scripts

### `templates/home.html`

- Purpose: public landing page
- Explains two-role workflow
- Highlights:
  - doctor-created patient IDs
  - live waveform during recording
  - default doctor login

### `templates/login.html`

- Purpose: shared login page for both roles
- Behavior:
  - doctor and patient wording changes based on `role`
  - doctor page shows default local login note

### `templates/register.html`

- Purpose: old patient self-registration form
- Current state: orphaned template
- Reason:
  - no active route renders this template
  - `/register` in `app.py` now redirects away instead of rendering registration

### `templates/doctor_dashboard.html`

- Purpose: main doctor control panel
- Features:
  - create patient account form
  - patient list table
  - summary stat cards
  - recent report feed
  - delete patient actions

### `templates/patient_dashboard.html`

- Purpose: patient self-service dashboard
- Features:
  - upload or browser recording form
  - latest saved report
  - report history
  - includes `static/patient.js`

### `templates/patient_detail.html`

- Purpose: doctor view for one patient
- Features:
  - doctor-side upload/record form for that patient
  - patient metadata
  - saved report history
  - delete patient action
  - includes `static/patient.js`

### `templates/_report_card.html`

- Purpose: reusable partial for showing a report
- Displays:
  - label and badges
  - summary text
  - quality flags
  - healthy/PD probabilities
  - segment count
  - stability
  - quality status
  - per-segment votes
  - skipped segment reasons
  - extracted features

## Static Files

### `static/styles.css`

- Purpose: full visual design system for the portal
- Main styling themes:
  - warm light background with teal/orange accents
  - glassmorphism-like cards
  - responsive grid layout
  - reusable button/badge/card/table styles
- Covers:
  - header
  - hero sections
  - auth pages
  - forms
  - recorder box
  - waveform canvas
  - inline status messages
  - report cards
  - metric tiles
  - tables
  - responsive breakpoints

### `static/patient.js`

- Purpose: client-side audio recording, preview, waveform rendering, quality pre-checks, and form submission
- Main responsibilities:
  - connect DOM controls
  - record from microphone
  - draw live waveform while recording
  - convert browser recording to WAV
  - preview recorded audio
  - inspect client-side duration/RMS/clipping/sample-rate
  - upload file/blob via `fetch`
  - refresh page after successful save

Client-side recording settings:

- target sample rate: `44100`
- max recording length: `10000 ms`
- minimum client duration: `6 seconds`
- minimum client RMS: `0.003`
- max client clipped fraction: `0.01`

Important design point:

- browser checks are only pre-checks; the server still performs final quality validation

## Data Files

### `data/parkinsons.data`

- Purpose: training data source
- Format: CSV-like text with header row
- Used by: `train_model.py`
- Not used directly by runtime inference in `app.py`

### `data/portal.db`

- Purpose: live application database
- Used by: `app.py`
- Main entities:
  - `users`
  - `reports`

`users` table fields:

- `id`
- `full_name`
- `mobile_number`
- `username`
- `password_hash`
- `role`
- `created_at`

`reports` table fields:

- `id`
- `patient_id`
- `source_filename`
- `prediction`
- `label`
- `healthy_probability`
- `parkinsons_probability`
- `confidence_gap`
- `stability`
- `segment_count`
- `summary`
- `features_json`
- `segment_results_json`
- `skipped_segments_json`
- `quality_status`
- `needs_retake`
- `quality_flags_json`
- `created_at`

## Model Files

### `models/model_metadata.json`

- Purpose: human-readable summary of the active trained model
- Runtime uses:
  - `selected_features`
  - `decision_threshold`
- Current active settings from metadata:
  - best candidate: `nonlinear_5_extra_trees`
  - classifier: `ExtraTreesClassifier`
  - feature set name: `nonlinear_5`
  - selected features: `HNR`, `RPDE`, `DFA`, `D2`, `PPE`
  - decision threshold: `0.675`

Reported subject-level out-of-fold metrics:

- balanced accuracy: `0.7708`
- PD recall/sensitivity: `0.9167`
- healthy recall/specificity: `0.6250`
- F1 score: `0.8980`
- ROC AUC: `0.8490`
- average precision: `0.9438`

Important note:

- the `evaluation_artifacts` paths inside this JSON are Unix absolute paths from the training environment, not Windows repository-relative paths
- current runtime does not depend on those JSON path strings, so the mismatch is mostly a documentation portability issue

### `models/prediction_pipeline.pkl`

- Purpose: active inference artifact
- Inferred structure:
  - scikit-learn `Pipeline`
  - includes `StandardScaler`
  - includes classifier stage
- Runtime status: primary model used by `app.py`

### `models/parkinsons_model.pkl`

- Purpose: legacy standalone classifier artifact
- Runtime status: fallback-only if `prediction_pipeline.pkl` is missing
- Training source: still written by current `train_model.py`

### `models/scaler.pkl`

- Purpose: legacy standalone scaler artifact
- Runtime status: fallback-only if `prediction_pipeline.pkl` is missing

### `models/rfe_selector.pkl`

- Purpose: legacy recursive feature elimination selector
- Evidence from artifact inspection:
  - serialized as scikit-learn `RFE`
  - base estimator is `LogisticRegression`
- Runtime status:
  - optional legacy fallback in `app.py`
- Important note:
  - current `train_model.py` no longer regenerates this file

### `models/selected_features.pkl`

- Purpose: serialized list of selected runtime features
- Current contents:
  - `HNR`
  - `RPDE`
  - `DFA`
  - `D2`
  - `PPE`
- Important note:
  - runtime actually prefers the same list from `model_metadata.json`

### `models/selected_mask.pkl`

- Purpose: legacy feature mask artifact from an older feature-selection workflow
- Artifact evidence:
  - stored as a NumPy boolean-array-like object with shape `(22,)`
- Runtime status:
  - not referenced anywhere in the current source code

## Evaluation Files

### `models/evaluation/recording_level_oof_predictions.csv`

- Purpose: out-of-fold predictions per recording
- Content includes:
  - recording name
  - subject ID
  - true label
  - class probabilities
  - fold
  - chosen candidate/model metadata

### `models/evaluation/subject_level_oof_predictions.csv`

- Purpose: out-of-fold predictions aggregated to the subject level
- Content includes:
  - averaged Parkinson's probability per subject
  - final threshold
  - chosen candidate/model info
  - final subject prediction and label

### `models/evaluation/subject_confusion_matrix.json`

- Purpose: machine-readable subject-level confusion matrix
- Current matrix:

```text
[[5, 3],
 [2, 22]]
```

Interpretation:

- true healthy predicted healthy: 5
- true healthy predicted Parkinson's: 3
- true Parkinson's predicted healthy: 2
- true Parkinson's predicted Parkinson's: 22

### `models/evaluation/subject_confusion_matrix.svg`

- Purpose: visual SVG rendering of the confusion matrix

### `models/evaluation/roc_curve.csv`

- Purpose: ROC curve numeric points

### `models/evaluation/roc_curve.svg`

- Purpose: ROC curve chart

### `models/evaluation/precision_recall_curve.csv`

- Purpose: precision-recall curve numeric points

### `models/evaluation/precision_recall_curve.svg`

- Purpose: precision-recall curve chart

## 6. Active Runtime Design Summary

The application is currently designed around this active runtime stack:

- Flask web portal
- SQLite local persistence
- browser-side WAV recording/upload
- server-side multi-segment voice analysis
- feature extraction producing 22 biomarkers
- active inference using 5 selected biomarkers:
  - `HNR`
  - `RPDE`
  - `DFA`
  - `D2`
  - `PPE`
- thresholded Parkinson's prediction with report quality assessment

## 7. Important Design Notes and Findings

### What is clearly active

- `app.py`
- `feature_extractor.py`
- `train_model.py`
- doctor/patient dashboard templates
- `static/patient.js`
- `static/styles.css`
- `data/parkinsons.data`
- `data/portal.db`
- `models/prediction_pipeline.pkl`
- `models/model_metadata.json`

### What is legacy or transitional

- `models/parkinsons_model.pkl`
- `models/scaler.pkl`
- `models/rfe_selector.pkl`
- `models/selected_mask.pkl`
- `/stream-chunk` endpoint
- `templates/register.html`

### Documentation or maintenance mismatches

- README still mentions patient registration, but self-registration is disabled in code
- `paper_6` feature-set name does not match its current 7-feature contents
- `robust_8` is defined but not evaluated
- `model_metadata.json` contains Unix absolute evaluation paths
- the checked-in `venv/` is a Linux-style environment and is not portable to this Windows workspace as-is
- dependency versions are not pinned, which can make pickled model loading fragile across machines

## 8. Short Architectural Conclusion

This project is a doctor-managed Parkinson's voice-screening portal built on Flask. The codebase combines three major concerns in one repository:

- web application and patient management
- scientific voice feature extraction
- machine-learning training and artifact generation

The strongest architectural center is `app.py`, which sits between the UI, SQLite, audio processing, and model prediction layers. The strongest scientific center is `feature_extractor.py`, and the strongest ML lifecycle center is `train_model.py`.

If this project is explained as a brain diagram, the simplest summary is:

```text
UI brain -> App brain -> Audio/Feature brain -> Model brain -> Database memory
```

That is the core structure of the repository.
