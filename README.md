# Parkinson Voice Screening Portal

Flask-based portal for Parkinson's voice screening with doctor-managed patient accounts, multi-segment inference, SQLite report storage, and a retraining workflow for improving the ML model over time.

## What This Project Contains

- doctor and patient login flows
- voice upload and in-browser recording
- multi-segment Parkinson's screening inference
- SQLite-based patient and report storage
- merged dataset training pipeline
- clinician feedback export flow for future retraining

## Recommended Environment

- Windows with PowerShell
- Python `3.11`
- `pip`
- Git

This project has been tested primarily with a local virtual environment in `.venv`.

## First-Time Setup on a New Laptop

After cloning the repository, change into the repository folder first, then run:

```powershell
cd Project_parkinsons
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want to recreate the virtual environment from scratch later, run this from the same folder:

```powershell
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set the joblib worker limit to avoid CPU-core detection warnings on some Windows systems:

```powershell
$env:LOKY_MAX_CPU_COUNT="8"
```

## Required Project Data

Before training, make sure these are present inside the project:

- `data\parkinsons.data`
- `data\HC_AH\`
- `data\PD_AH\`

If the repository clone does not include these files or folders, copy them into the `data` folder before continuing.

## Quick Start If Trained Model Already Exists

If the repository already contains these files:

- `models\prediction_pipeline.pkl`
- `models\selected_features.pkl`
- `models\model_metadata.json`

then you can run the app directly after installing requirements:

```powershell
.\.venv\Scripts\Activate.ps1
$env:LOKY_MAX_CPU_COUNT="8"
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Full First-Time Run From Clone

If you want to verify the environment, rebuild the merged dataset, retrain the best production model, and then run the app, use this order:

```powershell
.\.venv\Scripts\Activate.ps1
$env:LOKY_MAX_CPU_COUNT="8"
python verify_setup.py
python merge_datasets.py
python train_model.py --dataset aligned --source-filter all --fixed-candidate robust_8_extra_trees --resampler smote
python app.py
```

This is the current recommended production training command because it performed better than the paper-style candidates in grouped subject-level validation.

## Default Doctor Login

The app seeds one doctor account on first run:

- Username: `doctor`
- Password: `doctor123`

For deployment or shared usage, override these with environment variables:

- `DOCTOR_USERNAME`
- `DOCTOR_PASSWORD`
- `SECRET_KEY`

Example:

```powershell
$env:DOCTOR_USERNAME="doctor"
$env:DOCTOR_PASSWORD="change_this_password"
$env:SECRET_KEY="replace_with_a_long_random_secret"
python app.py
```

## Portal Workflow

### Doctor-managed patient creation

1. Log in as doctor.
2. Create a patient from the doctor dashboard using full name, mobile number, gender, and a doctor-set password.
3. The portal generates the patient ID automatically.
4. Share the generated patient ID and password with the patient.

### Patient screening flow

1. The patient logs in with the doctor-created credentials.
2. The patient uploads or records a sustained `"Ahh"` sample.
3. Recommended recording length is `8 to 10 seconds`.
4. The portal analyzes multiple overlapping segments and saves the report.
5. If quality is weak, the portal can still save the result and mark it as `retake recommended`.

### Doctor-side screening flow

Doctors can also open any patient record and upload or record a sample on the patient's behalf.

## Supported Upload Formats

The portal accepts:

- `WAV`
- `MP3`
- `M4A`
- `AAC`
- `OGG`
- `FLAC`
- `MP4`
- `WebM`

Notes:

- `WAV` is still the safest and most reliable format.
- Non-WAV uploads are normalized to temporary WAV on the server before segmentation.
- Short recordings may still be analyzed, but the report can be marked as `retake`.

## Important Generated Files

- `data\portal.db`
  - SQLite database created automatically by the app
- `models\prediction_pipeline.pkl`
  - active inference model
- `models\selected_features.pkl`
  - selected feature list for inference
- `models\model_metadata.json`
  - saved metrics, threshold, and training configuration
- `models\evaluation\`
  - out-of-fold predictions and evaluation artifacts

## How To Check the Current Active Model

```powershell
Get-Content models\model_metadata.json
```

Look for:

- `fixed_candidate`
- `resampler`
- `decision_threshold`
- `subject_level_oof_metrics`

## Training and Accuracy Improvement

### Recommended production retraining

```powershell
python train_model.py --dataset aligned --source-filter all --fixed-candidate robust_8_extra_trees --resampler smote
```

### Paper-inspired experiments

```powershell
python train_model.py --dataset aligned --source-filter all --fixed-candidate robust_8_extra_trees --resampler svmsmote
python train_model.py --dataset aligned --source-filter all --fixed-candidate paper_15_extra_trees --resampler smote
python train_model.py --dataset aligned --source-filter all --fixed-candidate paper_15_extra_trees --resampler svmsmote
python train_model.py --dataset aligned --source-filter all --fixed-candidate paper_15_xgboost --resampler svmsmote
```

Paper-style notes:

- `paper_15_xgboost` requires `xgboost`, which is already listed in `requirements.txt`.
- `shap` is also included in `requirements.txt` for future explainability work.
- The paper-style models are currently for comparison, not the default production choice.

## Verify Environment and Dataset

Run this after setup if you want to confirm everything is available:

```powershell
python verify_setup.py
```

It checks:

- required dataset files
- WAV folders
- key Python packages
- feature extraction on a sample file
- merged row expectations

## Data Collection and Feedback Loop

The portal supports a consent-based retraining workflow:

1. When saving a report, patient or doctor can consent to anonymised training use.
2. Doctor reviews saved reports.
3. Doctor sets clinician confirmation label:
   - `confirmed_pd`
   - `confirmed_healthy`
   - `unconfirmed`
4. Export confirmed, consented records with:

```powershell
GET /api/export-training-data
```

The export payload is a JSON list of `{features, label}` items.

## Dataset Augmentation

Use the augmentation helper to expand a WAV dataset:

```powershell
python augment_dataset.py --input-dir data\raw_recordings --output-dir data\augmented
```

Preview only:

```powershell
python augment_dataset.py --input-dir data\raw_recordings --output-dir data\augmented --dry-run
```

## Active Routes

### HTML routes

- `GET /`
- `GET|POST /register`
- `GET|POST /login/patient`
- `GET|POST /login/doctor`
- `GET /logout`
- `GET /patient/dashboard`
- `GET /doctor/dashboard`
- `GET /doctor/patient/<patient_id>`
- `POST /doctor/patient/<patient_id>/delete`
- `POST /doctor/patients/create`
- `POST /doctor/patient/<patient_id>/reports`

### JSON/API routes

- `GET /health`
- `POST /analyze-voice`
- `POST /stream-chunk`
- `POST /api/reports`
- `PATCH /api/reports/<report_id>/confirm`
- `GET /api/export-training-data`

## Troubleshooting

### `ModuleNotFoundError`

You are probably using the wrong Python interpreter. Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

This project currently targets Python `3.11`. On Windows, the safest way to create the environment is:

```powershell
py -3.11 -m venv .venv
```

If you create the environment with Python `3.13`, packages such as `numpy==1.26.4` may try to build from source and fail with compiler errors like `Unknown compiler(s)` or missing `cl`.

### `Primary model not found at models/prediction_pipeline.pkl`

Train the production model first:

```powershell
python train_model.py --dataset aligned --source-filter all --fixed-candidate robust_8_extra_trees --resampler smote
```

### Upload fails for some audio files

- Try `WAV` first.
- If the file is very short, clipped, or unstable, the app may request a retake.
- Recommended voice sample: steady `"Ahh"` for `8 to 10 seconds` in a quiet room.

### `Could not find the number of physical cores`

Set:

```powershell
$env:LOKY_MAX_CPU_COUNT="8"
```

## Project Reference Notes

- Current project brain/report file: `PROJECT_BRAIN_STRUCTURE_REPORT_CURRENT.md`
- Main application entry point: `app.py`
- Training pipeline: `train_model.py`
- Merge pipeline: `merge_datasets.py`
- Verification helper: `verify_setup.py`

cd D:\New folder\Project_parkinsons

pow shell
cd "D:\New folder\Project_parkinsons"
& "..\.venv\Scripts\Activate.ps1"
$env:LOKY_MAX_CPU_COUNT="8"
python app.py
