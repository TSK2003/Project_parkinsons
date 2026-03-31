# Parkinson Voice Screening Portal

Flask-based portal for Parkinson's voice screening with doctor-managed patient accounts, multi-segment inference, SQLite report storage, and a retraining/export workflow for improving the ML model over time.

## Requirements

- Python 3.11 recommended
- `pip` and a virtual environment tool
- A trained model artifact at `models/prediction_pipeline.pkl`

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:LOKY_MAX_CPU_COUNT=8
python verify_setup.py
python merge_datasets.py
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000`.

## Default Doctor Login

The app seeds one doctor account on first run:

- Username: `doctor`
- Password: `doctor123`

Override these in deployment with:

- `DOCTOR_USERNAME`
- `DOCTOR_PASSWORD`
- `SECRET_KEY`

## Workflow

### Doctor-managed patient creation

1. Log in as a doctor.
2. Create a patient from the doctor dashboard using full name, mobile number, gender, and a doctor-set password.
3. The portal generates the patient ID automatically.
4. Share the generated patient ID and password with the patient.

### Patient screening flow

1. The patient logs in with the doctor-created credentials.
2. The patient uploads or records an 8 to 10 second sustained "Ahh".
3. The portal analyzes multiple overlapping segments and saves the report.
4. The doctor can review the report, biomarkers, and clinician confirmation status from the patient detail page.

### Doctor-side screening flow

Doctors can also open any patient record and upload or record a voice sample on the patient's behalf.

## Active Endpoints

### HTML routes

- `GET /` - landing page
- `GET|POST /register` - legacy route that redirects to patient login
- `GET|POST /login/patient` - patient login
- `GET|POST /login/doctor` - doctor login
- `GET /logout` - clear session
- `GET /patient/dashboard` - patient dashboard and report history
- `GET /doctor/dashboard` - doctor dashboard and patient list
- `GET /doctor/patient/<patient_id>` - patient detail page
- `POST /doctor/patient/<patient_id>/delete` - delete patient and reports
- `POST /doctor/patients/create` - create patient account
- `POST /doctor/patient/<patient_id>/reports` - doctor uploads or records a report for a patient

### JSON/API routes

- `GET /health` - health check
- `POST /analyze-voice` - analyze an uploaded file without saving
- `POST /stream-chunk` - single-chunk inference endpoint
- `POST /api/reports` - patient creates and saves a report
- `PATCH /api/reports/<report_id>/confirm` - doctor confirms a saved report label
- `GET /api/export-training-data` - doctor exports consented and clinician-confirmed records for retraining

## Retraining

Run the training pipeline whenever you update the dataset or want to regenerate the inference model:

```bash
python train_model.py
```

The script:

- evaluates multiple candidate feature sets and classifiers
- applies SMOTE inside cross-validation training folds
- tunes the decision threshold for balanced accuracy with a PD recall floor
- writes `models/prediction_pipeline.pkl`
- writes `models/model_metadata.json`

## Data Collection

The portal supports a consent-based feedback loop for future retraining:

1. When a patient or doctor saves a report, they can mark that recording as consented for anonymised training use.
2. A doctor reviews saved reports from the patient detail page.
3. The doctor sets a clinician confirmation label: `confirmed_pd`, `confirmed_healthy`, or `unconfirmed`.
4. `/api/export-training-data` exports only records where:
   - `consented_for_training = 1`
   - `clinician_confirmed_label` is `confirmed_pd` or `confirmed_healthy`

The export payload is a JSON list of `{features, label}` items ready for downstream dataset assembly.

## Dataset Augmentation

Use the augmentation helper to expand a WAV dataset with pitch shifts, time stretching, and controlled noise:

```bash
python augment_dataset.py --input-dir data\raw_recordings --output-dir data\augmented
```

Use `--dry-run` to preview generated filenames without writing files.
