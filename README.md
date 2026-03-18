# Parkinson Voice Portal

This project is now a Flask web portal with:

- patient registration and login
- doctor login
- saved patient reports in SQLite
- doctor-side patient list and report review
- improved voice analysis using multi-segment aggregation and stricter audio quality checks

## Run the project

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Flask app:

```bash
python3 app.py
```

4. Open:

```text
http://127.0.0.1:5000
```

## Default local login

The app seeds one doctor account on first run:

- Username: `doctor`
- Password: `doctor123`

For deployment, change these with environment variables:

- `DOCTOR_USERNAME`
- `DOCTOR_PASSWORD`
- `SECRET_KEY`

## Patient flow

1. Register a patient account.
2. Log in as that patient.
3. Upload an audio file or record 8 to 10 seconds of a steady `Ahh`.
4. Save the generated report.

## Doctor flow

1. Log in with the doctor account.
2. Open the patient list.
3. Review each patient's saved reports and biomarkers.

## API endpoints

- `GET /health` - portal health check
- `POST /analyze-voice` - analyze an uploaded audio file without saving
- `POST /stream-chunk` - legacy single-chunk prediction endpoint
- `POST /api/reports` - patient-only endpoint that analyzes audio and saves a report

## Retrain the model

Run:

```bash
python3 train_model.py
```

The training script now compares multiple classifiers with cross-validation, saves the best inference pipeline, and writes artifacts to `models/`.
