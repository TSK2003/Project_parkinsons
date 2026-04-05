import json
import os
import sqlite3
import tempfile
from datetime import datetime
from functools import wraps

import joblib
import librosa
import numpy as np
import pandas as pd
import sklearn
from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_cors import CORS
from scipy.io import wavfile
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from feature_extractor import ALL_FEATURE_NAMES, extract_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "portal.db")

MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "prediction_pipeline.pkl")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")

ALL_FEATURES = list(ALL_FEATURE_NAMES)
ALLOWED_GENDERS = {"male", "female", "unspecified"}
ALLOWED_CLINICIAN_LABELS = {"confirmed_pd", "confirmed_healthy", "unconfirmed"}
SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
    ".webm",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "parkinsons-portal-dev-key")
app.config["DATABASE_PATH"] = DATABASE_PATH
app.config["SEGMENT_SECONDS"] = float(os.environ.get("SEGMENT_SECONDS", "3.0"))
app.config["SEGMENT_OVERLAP_SECONDS"] = float(
    os.environ.get("SEGMENT_OVERLAP_SECONDS", "1.0")
)
app.config["MIN_VALID_SEGMENTS"] = int(os.environ.get("MIN_VALID_SEGMENTS", "3"))
app.config["MAX_SKIPPED_SEGMENT_RATIO"] = float(
    os.environ.get("MAX_SKIPPED_SEGMENT_RATIO", "0.34")
)
app.config["MIN_SEGMENT_STABILITY"] = float(
    os.environ.get("MIN_SEGMENT_STABILITY", "0.70")
)
app.config["MIN_CONFIDENCE_GAP"] = float(
    os.environ.get("MIN_CONFIDENCE_GAP", "0.18")
)
CORS(app)

model_metadata = {}
selected_model_features = ALL_FEATURES
prediction_threshold = 0.5
uses_gender_specific_extraction = False


def force_single_thread_inference(estimator) -> None:
    if hasattr(estimator, "steps"):
        for _, step in estimator.steps:
            force_single_thread_inference(step)

    if hasattr(estimator, "n_jobs"):
        try:
            estimator.set_params(n_jobs=1)
        except Exception:
            estimator.n_jobs = 1

if os.path.exists(MODEL_METADATA_PATH):
    with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as handle:
        model_metadata = json.load(handle)
    selected_model_features = model_metadata.get("selected_features", ALL_FEATURES)
    prediction_threshold = float(model_metadata.get("decision_threshold", 0.5))
    uses_gender_specific_extraction = bool(
        model_metadata.get("uses_gender_specific_extraction", False)
    )

if not os.path.exists(MODEL_PIPELINE_PATH):
    raise FileNotFoundError(
        f"Primary model not found at {MODEL_PIPELINE_PATH}. "
        "Run `python train_model.py` to generate it."
    )

prediction_pipeline = joblib.load(MODEL_PIPELINE_PATH)
force_single_thread_inference(prediction_pipeline)

trained_sklearn = model_metadata.get("sklearn_version")
current_sklearn = sklearn.__version__
if trained_sklearn and trained_sklearn != current_sklearn:
    app.logger.warning(
        f"Model was trained with scikit-learn {trained_sklearn} "
        f"but current version is {current_sklearn}. "
        "Consider retraining to avoid compatibility issues."
    )


def utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_gender(gender: str | None) -> str:
    if gender is None:
        return "unspecified"
    cleaned = gender.strip().lower()
    return cleaned if cleaned in ALLOWED_GENDERS else "unspecified"


def normalize_clinician_label(label: str | None) -> str:
    if label is None:
        return "unconfirmed"
    cleaned = label.strip().lower()
    return cleaned if cleaned in ALLOWED_CLINICIAN_LABELS else ""


def ensure_supported_audio_file(filename: str) -> None:
    _, ext = os.path.splitext(filename or "")
    suffix = ext.lower() or ".wav"
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format '{suffix}'. Please upload one of: {allowed}."
        )


def normalize_audio_for_analysis(audio_path: str) -> tuple[str, list[str]]:
    _, ext = os.path.splitext(audio_path)
    suffix = ext.lower() or ".wav"

    if suffix == ".wav":
        return audio_path, []

    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format '{suffix}'. Please upload one of: {allowed}."
        )

    try:
        audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    except Exception as exc:
        raise ValueError(
            "The uploaded audio could not be decoded. Please try WAV, MP3, M4A, AAC, OGG, FLAC, MP4, or WebM."
        ) from exc

    if sample_rate is None or int(sample_rate) <= 0:
        raise ValueError("The uploaded audio file does not have a valid sample rate.")

    if getattr(audio_data, "ndim", 1) > 1:
        audio_data = np.mean(audio_data, axis=0)

    audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.size == 0 or not np.isfinite(audio_data).all():
        raise ValueError("The uploaded audio file could not be normalized for analysis.")

    audio_data = np.clip(audio_data, -1.0, 1.0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wavfile.write(temp_file.name, int(sample_rate), audio_data)
        return temp_file.name, [temp_file.name]


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        os.makedirs(DATA_DIR, exist_ok=True)
        connection = sqlite3.connect(app.config["DATABASE_PATH"])
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


@app.teardown_appcontext
def close_db(_exception) -> None:
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db() -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            mobile_number TEXT,
            gender TEXT NOT NULL DEFAULT 'unspecified',
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            role TEXT NOT NULL CHECK(role IN ('patient', 'doctor')),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            source_filename TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            label TEXT NOT NULL,
            healthy_probability REAL NOT NULL,
            parkinsons_probability REAL NOT NULL,
            confidence_gap REAL NOT NULL,
            stability REAL NOT NULL,
            segment_count INTEGER NOT NULL,
            summary TEXT NOT NULL,
            features_json TEXT NOT NULL,
            segment_results_json TEXT NOT NULL,
            skipped_segments_json TEXT NOT NULL DEFAULT '[]',
            quality_status TEXT NOT NULL DEFAULT 'accepted',
            needs_retake INTEGER NOT NULL DEFAULT 0,
            quality_flags_json TEXT NOT NULL DEFAULT '[]',
            consented_for_training INTEGER NOT NULL DEFAULT 0,
            clinician_confirmed_label TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    ensure_user_columns()
    ensure_report_columns()
    db.commit()
    ensure_default_doctor()


def ensure_user_columns() -> None:
    db = get_db()
    columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(users)").fetchall()
    }
    if "mobile_number" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN mobile_number TEXT")
    if "gender" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN gender TEXT DEFAULT 'unspecified'")


def ensure_report_columns() -> None:
    db = get_db()
    columns = {
        row["name"]
        for row in db.execute("PRAGMA table_info(reports)").fetchall()
    }
    missing_columns = [
        (
            "quality_status",
            "TEXT NOT NULL DEFAULT 'accepted'",
        ),
        (
            "needs_retake",
            "INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "quality_flags_json",
            "TEXT NOT NULL DEFAULT '[]'",
        ),
        (
            "consented_for_training",
            "INTEGER NOT NULL DEFAULT 0",
        ),
        (
            "clinician_confirmed_label",
            "TEXT",
        ),
    ]
    for column_name, column_sql in missing_columns:
        if column_name not in columns:
            db.execute(f"ALTER TABLE reports ADD COLUMN {column_name} {column_sql}")


def ensure_default_doctor() -> None:
    db = get_db()
    existing_doctor = db.execute(
        "SELECT id FROM users WHERE role = 'doctor' LIMIT 1"
    ).fetchone()
    if existing_doctor is not None:
        return

    doctor_username = os.environ.get("DOCTOR_USERNAME", "doctor")
    doctor_password = os.environ.get("DOCTOR_PASSWORD", "doctor123")
    db.execute(
        """
        INSERT INTO users (full_name, mobile_number, gender, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, 'doctor', ?)
        """,
        (
            "Dr. Portal",
            None,
            "unspecified",
            doctor_username,
            generate_password_hash(doctor_password),
            utc_now(),
        ),
    )
    db.commit()


def row_to_user(row):
    return dict(row) if row is not None else None


def get_user_by_id(user_id: int, role: str | None = None):
    if user_id is None:
        return None
    db = get_db()
    if role is None:
        row = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    else:
        row = db.execute(
            "SELECT * FROM users WHERE id = ? AND role = ?", (user_id, role)
        ).fetchone()
    return row_to_user(row)


def get_user_by_username(username: str, role: str | None = None):
    db = get_db()
    if role is None:
        row = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    else:
        row = db.execute(
            "SELECT * FROM users WHERE username = ? AND role = ?",
            (username, role),
        ).fetchone()
    return row_to_user(row)


def hydrate_report(row):
    if row is None:
        return None

    report = dict(row)
    report["features"] = json.loads(report.pop("features_json") or "{}")
    report["segment_results"] = json.loads(report.pop("segment_results_json") or "[]")
    report["skipped_segments"] = json.loads(
        report.pop("skipped_segments_json") or "[]"
    )
    report["quality_flags"] = json.loads(report.pop("quality_flags_json", "[]") or "[]")
    report["quality_status"] = report.get("quality_status") or "accepted"
    report["needs_retake"] = bool(report.get("needs_retake", 0))
    report["consented_for_training"] = bool(report.get("consented_for_training", 0))
    report["clinician_confirmed_label"] = (
        report.get("clinician_confirmed_label") or "unconfirmed"
    )
    return report


def get_report_by_id(report_id: int):
    db = get_db()
    row = db.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return hydrate_report(row)


def list_reports_for_patient(patient_id: int):
    db = get_db()
    rows = db.execute(
        """
        SELECT *
        FROM reports
        WHERE patient_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (patient_id,),
    ).fetchall()
    return [hydrate_report(row) for row in rows]


def list_recent_reports(limit: int = 8):
    db = get_db()
    rows = db.execute(
        """
        SELECT
            reports.*,
            users.full_name AS patient_name,
            users.username AS patient_username
        FROM reports
        JOIN users ON users.id = reports.patient_id
        ORDER BY reports.created_at DESC, reports.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    recent_reports = []
    for row in rows:
        item = hydrate_report(row)
        recent_reports.append(item)
    return recent_reports


def list_patient_summaries():
    db = get_db()
    rows = db.execute(
        """
        SELECT
            users.id,
            users.full_name,
            users.mobile_number,
            users.gender,
            users.username,
            users.created_at,
            COUNT(reports.id) AS report_count,
            MAX(reports.created_at) AS latest_report_at,
            (
                SELECT label
                FROM reports AS latest
                WHERE latest.patient_id = users.id
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            ) AS latest_label,
            (
                SELECT parkinsons_probability
                FROM reports AS latest
                WHERE latest.patient_id = users.id
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            ) AS latest_pd_probability,
            (
                SELECT stability
                FROM reports AS latest
                WHERE latest.patient_id = users.id
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            ) AS latest_stability
        FROM users
        LEFT JOIN reports ON reports.patient_id = users.id
        WHERE users.role = 'patient'
        GROUP BY users.id
        ORDER BY (MAX(reports.created_at) IS NULL), MAX(reports.created_at) DESC, users.full_name ASC
        """
    ).fetchall()
    return [dict(row) for row in rows]


def normalize_mobile_number(mobile_number: str) -> str:
    return "".join(character for character in mobile_number if character.isdigit())


def generate_patient_identifier(full_name: str, mobile_number: str) -> str:
    cleaned_name = "".join(
        character for character in full_name.lower() if character.isalnum()
    )
    prefix = (cleaned_name[:4] or "patn").ljust(4, "x")
    return f"{prefix}{mobile_number[-4:]}"


def create_patient_record(
    full_name: str,
    mobile_number: str,
    password: str,
    gender: str = "unspecified",
) -> dict:
    cleaned_name = " ".join(full_name.split())
    if not cleaned_name:
        raise ValueError("Patient name is required.")

    normalized_mobile = normalize_mobile_number(mobile_number)
    if len(normalized_mobile) < 10:
        raise ValueError("Mobile number must contain at least 10 digits.")
    if len(password) < 6:
        raise ValueError("Patient password must be at least 6 characters long.")
    normalized_gender = normalize_gender(gender)

    patient_identifier = generate_patient_identifier(cleaned_name, normalized_mobile)
    db = get_db()
    existing_patient = db.execute(
        "SELECT id FROM users WHERE username = ? LIMIT 1",
        (patient_identifier,),
    ).fetchone()
    if existing_patient is not None:
        raise ValueError(
            "That generated patient ID already exists. Please verify the name and mobile number."
        )

    cursor = db.execute(
        """
        INSERT INTO users (full_name, mobile_number, gender, username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?, 'patient', ?)
        """,
        (
            cleaned_name,
            normalized_mobile,
            normalized_gender,
            patient_identifier,
            generate_password_hash(password),
            utc_now(),
        ),
    )
    db.commit()
    return get_user_by_id(cursor.lastrowid, role="patient")


def delete_patient_record(patient_id: int) -> dict | None:
    patient = get_user_by_id(patient_id, role="patient")
    if patient is None:
        return None

    db = get_db()
    db.execute("DELETE FROM users WHERE id = ? AND role = 'patient'", (patient_id,))
    db.commit()
    return patient


def save_report(
    patient_id: int,
    source_filename: str,
    report_data: dict,
    consented_for_training: bool = False,
):
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO reports (
            patient_id,
            source_filename,
            prediction,
            label,
            healthy_probability,
            parkinsons_probability,
            confidence_gap,
            stability,
            segment_count,
            summary,
            features_json,
            segment_results_json,
            skipped_segments_json,
            quality_status,
            needs_retake,
            quality_flags_json,
            consented_for_training,
            clinician_confirmed_label,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            source_filename,
            report_data["prediction"],
            report_data["label"],
            report_data["probabilities"]["healthy"],
            report_data["probabilities"]["parkinsons"],
            report_data["confidence_gap"],
            report_data["stability"],
            report_data["segment_count"],
            report_data["summary"],
            json.dumps(report_data["features"]),
            json.dumps(report_data["segment_results"]),
            json.dumps(report_data["skipped_segments"]),
            report_data["quality_status"],
            int(report_data["needs_retake"]),
            json.dumps(report_data["quality_flags"]),
            int(bool(consented_for_training)),
            "unconfirmed",
            utc_now(),
        ),
    )
    db.commit()
    return get_report_by_id(cursor.lastrowid)


@app.before_request
def load_current_user() -> None:
    user_id = session.get("user_id")
    g.user = get_user_by_id(user_id)
    if user_id and g.user is None:
        session.clear()


@app.context_processor
def inject_template_helpers():
    return {
        "current_user": g.get("user"),
    }


@app.template_filter("pct")
def pct_filter(value):
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


@app.template_filter("dt")
def dt_filter(value):
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %I:%M %p")
    except ValueError:
        return value


def login_required(role: str | None = None):
    def decorator(view):
        @wraps(view)
        def wrapped_view(*args, **kwargs):
            if g.user is None:
                flash("Please log in to continue.", "error")
                return redirect(url_for("home"))

            if role is not None and g.user["role"] != role:
                flash("You do not have access to that page.", "error")
                if g.user["role"] == "doctor":
                    return redirect(url_for("doctor_dashboard"))
                return redirect(url_for("patient_dashboard"))

            return view(*args, **kwargs)

        return wrapped_view

    return decorator


def infer_prediction(feature_frame):
    model_frame = feature_frame[selected_model_features]
    probabilities = prediction_pipeline.predict_proba(model_frame)[0]
    prediction = int(float(probabilities[1]) >= prediction_threshold)
    return prediction, probabilities


def run_prediction(audio_path: str, gender: str | None = None) -> dict:
    cleanup_paths = []
    try:
        normalized_audio_path, cleanup_paths = normalize_audio_for_analysis(audio_path)
        extraction_gender = normalize_gender(gender) if uses_gender_specific_extraction else None
        features_dict = extract_features(normalized_audio_path, gender=extraction_gender)
        feature_frame = pd.DataFrame(
            [{name: features_dict[name] for name in ALL_FEATURES}],
            columns=ALL_FEATURES,
        )
        prediction, probabilities = infer_prediction(feature_frame)

        healthy_probability = float(probabilities[0])
        parkinsons_probability = float(probabilities[1])
        return {
            "prediction": prediction,
            "label": "Parkinson's Detected" if prediction == 1 else "Healthy",
            "probabilities": {
                "healthy": round(healthy_probability, 4),
                "parkinsons": round(parkinsons_probability, 4),
            },
            "confidence_gap": round(abs(parkinsons_probability - healthy_probability), 4),
            "features": features_dict,
        }
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)


def split_audio_segments(audio_path: str):
    sample_rate, data = wavfile.read(audio_path)

    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)

    segment_seconds = app.config["SEGMENT_SECONDS"]
    overlap_seconds = app.config["SEGMENT_OVERLAP_SECONDS"]
    segment_samples = max(int(sample_rate * segment_seconds), 1)
    step_samples = max(int(sample_rate * max(segment_seconds - overlap_seconds, 0.5)), 1)

    if len(data) <= segment_samples:
        return [audio_path], []

    starts = list(range(0, len(data) - segment_samples + 1, step_samples))
    final_start = len(data) - segment_samples
    if starts and final_start - starts[-1] > step_samples // 2:
        starts.append(final_start)

    temp_paths = []
    for start in starts:
        segment = data[start : start + segment_samples]
        if np.issubdtype(segment.dtype, np.floating):
            segment = segment.astype(np.float32)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            wavfile.write(temp_file.name, sample_rate, segment)
            temp_paths.append(temp_file.name)

    return temp_paths, temp_paths.copy()


def build_report_summary(
    prediction: int,
    healthy_probability: float,
    parkinsons_probability: float,
    stability: float,
    valid_segments: int,
    skipped_segments: list[dict],
    quality_status: str,
    quality_flags: list[str],
) -> str:
    confidence = parkinsons_probability if prediction == 1 else healthy_probability
    strength = "high" if confidence >= 0.8 and stability >= 0.7 else "moderate"
    result_label = "Parkinson's pattern" if prediction == 1 else "healthy pattern"
    skipped_note = ""
    if skipped_segments:
        skipped_note = f" {len(skipped_segments)} low-quality segment(s) were skipped."

    quality_note = ""
    if quality_status == "retake":
        quality_note = (
            " Recording quality was not strong enough for a reliable screen. "
            "Please retake the sample before relying on this result."
        )
    elif quality_status == "uncertain":
        quality_note = " This screen is low confidence. A repeat recording is recommended."
    if quality_flags:
        quality_note += " Reasons: " + "; ".join(quality_flags) + "."

    return (
        f"Aggregated {valid_segments} valid voice segment(s). "
        f"Final result suggests a {result_label} with {confidence * 100:.1f}% confidence "
        f"and {stability * 100:.1f}% segment agreement ({strength} stability)."
        f"{skipped_note}"
        f"{quality_note}"
    )


def assess_report_quality(
    stability: float,
    confidence_gap: float,
    valid_segments: int,
    skipped_segments: list[dict],
) -> dict:
    total_segments = valid_segments + len(skipped_segments)
    skipped_ratio = len(skipped_segments) / max(total_segments, 1)
    quality_flags = []

    if valid_segments < app.config["MIN_VALID_SEGMENTS"]:
        quality_flags.append(
            "Too few valid voice windows were usable for a stable decision"
        )
    if skipped_ratio > app.config["MAX_SKIPPED_SEGMENT_RATIO"]:
        quality_flags.append(
            "Too many low-quality windows were skipped from the recording"
        )
    if stability < app.config["MIN_SEGMENT_STABILITY"]:
        quality_flags.append(
            "Segment votes were inconsistent across the sample"
        )
    if confidence_gap < app.config["MIN_CONFIDENCE_GAP"]:
        quality_flags.append(
            "Healthy and Parkinson's probabilities were too close together"
        )

    if (
        valid_segments < app.config["MIN_VALID_SEGMENTS"]
        or skipped_ratio > app.config["MAX_SKIPPED_SEGMENT_RATIO"]
    ):
        quality_status = "retake"
    elif quality_flags:
        quality_status = "uncertain"
    else:
        quality_status = "accepted"

    return {
        "quality_status": quality_status,
        "needs_retake": quality_status != "accepted",
        "quality_flags": quality_flags,
        "skipped_segment_ratio": round(skipped_ratio, 4),
        "total_segment_count": total_segments,
    }


def analyze_voice_sample(audio_path: str, gender: str | None = None) -> dict:
    cleanup_paths = []
    segment_results = []
    skipped_segments = []

    try:
        normalized_audio_path, normalization_cleanup_paths = normalize_audio_for_analysis(audio_path)
        cleanup_paths.extend(normalization_cleanup_paths)
        segment_paths, segment_cleanup_paths = split_audio_segments(normalized_audio_path)
        cleanup_paths.extend(segment_cleanup_paths)

        for index, segment_path in enumerate(segment_paths, start=1):
            try:
                segment_result = run_prediction(segment_path, gender=gender)
                segment_result["segment_index"] = index
                segment_results.append(segment_result)
            except ValueError as exc:
                skipped_segments.append({"segment_index": index, "reason": str(exc)})

        if not segment_results:
            if skipped_segments:
                raise ValueError(skipped_segments[0]["reason"])
            raise ValueError("The uploaded audio could not be analyzed.")

        weights = [max(item.get("confidence_gap", 0.01), 0.01) for item in segment_results]
        parkinsons_probability = float(
            np.average(
                [item["probabilities"]["parkinsons"] for item in segment_results],
                weights=weights,
            )
        )
        healthy_probability = float(1.0 - parkinsons_probability)
        prediction = int(parkinsons_probability >= prediction_threshold)
        stability = float(
            np.mean([item["prediction"] == prediction for item in segment_results])
        )

        averaged_features = {
            name: round(
                float(np.mean([item["features"][name] for item in segment_results])), 5
            )
            for name in ALL_FEATURES
        }

        label = "Parkinson's Detected" if prediction == 1 else "Healthy"
        confidence_gap = abs(parkinsons_probability - healthy_probability)
        quality_assessment = assess_report_quality(
            stability=stability,
            confidence_gap=confidence_gap,
            valid_segments=len(segment_results),
            skipped_segments=skipped_segments,
        )
        summary = build_report_summary(
            prediction=prediction,
            healthy_probability=healthy_probability,
            parkinsons_probability=parkinsons_probability,
            stability=stability,
            valid_segments=len(segment_results),
            skipped_segments=skipped_segments,
            quality_status=quality_assessment["quality_status"],
            quality_flags=quality_assessment["quality_flags"],
        )

        return {
            "prediction": prediction,
            "label": label,
            "probabilities": {
                "healthy": round(healthy_probability, 4),
                "parkinsons": round(parkinsons_probability, 4),
            },
            "confidence_gap": round(confidence_gap, 4),
            "stability": round(stability, 4),
            "segment_count": len(segment_results),
            "summary": summary,
            "features": averaged_features,
            "segment_results": segment_results,
            "skipped_segments": skipped_segments,
            "quality_status": quality_assessment["quality_status"],
            "needs_retake": quality_assessment["needs_retake"],
            "quality_flags": quality_assessment["quality_flags"],
            "skipped_segment_ratio": quality_assessment["skipped_segment_ratio"],
            "total_segment_count": quality_assessment["total_segment_count"],
        }
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                os.unlink(path)


def file_suffix(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lower() or ".wav"


@app.route("/")
def home():
    if g.user is not None:
        if g.user["role"] == "doctor":
            return redirect(url_for("doctor_dashboard"))
        return redirect(url_for("patient_dashboard"))
    return render_template("home.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "message": "Doctor-managed patient portal with patient login is running.",
        }
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    flash(
        "Patient self-registration has been removed. Use the patient ID and password created by the doctor.",
        "info",
    )
    return redirect(url_for("patient_login"))


def process_login(role: str):
    if g.user is not None and g.user["role"] == role:
        if role == "doctor":
            return redirect(url_for("doctor_dashboard"))
        return redirect(url_for("patient_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_username(username, role=role)

        if (
            user is None
            or not user.get("password_hash")
            or not check_password_hash(user["password_hash"], password)
        ):
            flash("Invalid username or password.", "error")
        else:
            session.clear()
            session["user_id"] = user["id"]
            flash(f"Welcome back, {user['full_name']}.", "success")
            if role == "doctor":
                return redirect(url_for("doctor_dashboard"))
            return redirect(url_for("patient_dashboard"))

    return render_template("login.html", role=role)


@app.route("/login/patient", methods=["GET", "POST"])
def patient_login():
    return process_login("patient")


@app.route("/login/doctor", methods=["GET", "POST"])
def doctor_login():
    return process_login("doctor")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


@app.route("/patient/dashboard")
@login_required(role="patient")
def patient_dashboard():
    reports = list_reports_for_patient(g.user["id"])
    latest_report = reports[0] if reports else None
    return render_template(
        "patient_dashboard.html",
        latest_report=latest_report,
        reports=reports,
    )


@app.route("/doctor/patients/create", methods=["POST"])
@login_required(role="doctor")
def doctor_create_patient():
    full_name = request.form.get("full_name", "").strip()
    mobile_number = request.form.get("mobile_number", "").strip()
    gender = normalize_gender(request.form.get("gender"))
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not full_name or not mobile_number or not password:
        flash("Patient name, mobile number, and password are required.", "error")
        return redirect(url_for("doctor_dashboard"))

    if password != confirm_password:
        flash("Patient password and confirm password must match.", "error")
        return redirect(url_for("doctor_dashboard"))

    try:
        patient = create_patient_record(full_name, mobile_number, password, gender=gender)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("doctor_dashboard"))
    except sqlite3.IntegrityError:
        flash(
            "Could not create the patient because that generated ID already exists.",
            "error",
        )
        return redirect(url_for("doctor_dashboard"))

    flash(
        (
            f"Patient created successfully. Patient ID: {patient['username']}. "
            "The patient can now log in with this ID and the password you created."
        ),
        "success",
    )
    return redirect(url_for("doctor_patient_detail", patient_id=patient["id"]))


@app.route("/doctor/dashboard")
@login_required(role="doctor")
def doctor_dashboard():
    patient_summaries = list_patient_summaries()
    recent_reports = list_recent_reports(limit=10)
    total_reports = sum(item["report_count"] for item in patient_summaries)
    return render_template(
        "doctor_dashboard.html",
        patient_summaries=patient_summaries,
        recent_reports=recent_reports,
        total_reports=total_reports,
    )


@app.route("/doctor/patient/<int:patient_id>")
@login_required(role="doctor")
def doctor_patient_detail(patient_id: int):
    patient = get_user_by_id(patient_id, role="patient")
    if patient is None:
        abort(404)

    reports = list_reports_for_patient(patient_id)
    return render_template("patient_detail.html", patient=patient, reports=reports)


@app.route("/doctor/patient/<int:patient_id>/delete", methods=["POST"])
@login_required(role="doctor")
def doctor_delete_patient(patient_id: int):
    patient = delete_patient_record(patient_id)
    if patient is None:
        flash("Patient account was not found.", "error")
    else:
        flash(
            f"Patient {patient['full_name']} ({patient['username']}) and all saved reports were removed.",
            "success",
        )
    return redirect(url_for("doctor_dashboard"))


@app.route("/doctor/patient/<int:patient_id>/reports", methods=["POST"])
@login_required(role="doctor")
def doctor_create_patient_report(patient_id: int):
    patient = get_user_by_id(patient_id, role="patient")
    if patient is None:
        return jsonify({"error": "Patient account was not found."}), 404

    if "audio" not in request.files:
        return jsonify({"error": "Please upload or record an audio sample first."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Please choose an audio file before analyzing."}), 400
    try:
        ensure_supported_audio_file(audio_file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    consented_for_training = request.form.get("consent_training") == "1"

    filename = secure_filename(audio_file.filename) or "voice_sample.wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(filename)) as temp:
        audio_file.save(temp.name)
        temp_path = temp.name

    try:
        report_data = analyze_voice_sample(temp_path, gender=patient.get("gender"))
        saved_report = save_report(
            patient_id,
            filename,
            report_data,
            consented_for_training=consented_for_training,
        )
        return jsonify(
            {
                "message": "Report generated and saved successfully.",
                "report": saved_report,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected analysis failure")
        return jsonify({"error": "Could not analyze the uploaded audio."}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route("/api/reports", methods=["POST"])
@login_required(role="patient")
def create_report():
    if "audio" not in request.files:
        return jsonify({"error": "Please upload or record an audio sample first."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Please choose an audio file before analyzing."}), 400
    try:
        ensure_supported_audio_file(audio_file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    consented_for_training = request.form.get("consent_training") == "1"

    filename = secure_filename(audio_file.filename) or "voice_sample.wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(filename)) as temp:
        audio_file.save(temp.name)
        temp_path = temp.name

    try:
        report_data = analyze_voice_sample(temp_path, gender=g.user.get("gender"))
        saved_report = save_report(
            g.user["id"],
            filename,
            report_data,
            consented_for_training=consented_for_training,
        )
        return jsonify(
            {
                "message": "Report generated and saved successfully.",
                "report": saved_report,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected analysis failure")
        return jsonify({"error": "Could not analyze the uploaded audio."}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route("/api/reports/<int:report_id>/confirm", methods=["PATCH"])
@login_required(role="doctor")
def confirm_report(report_id: int):
    report = get_report_by_id(report_id)
    if report is None:
        return jsonify({"error": "Report was not found."}), 404

    payload = request.get_json(silent=True) or request.form
    clinician_label = normalize_clinician_label(payload.get("clinician_confirmed_label"))
    if not clinician_label:
        return jsonify({"error": "Invalid clinician confirmation label."}), 400

    db = get_db()
    db.execute(
        "UPDATE reports SET clinician_confirmed_label = ? WHERE id = ?",
        (clinician_label, report_id),
    )
    db.commit()
    return jsonify(
        {
            "message": "Clinician confirmation updated successfully.",
            "report": get_report_by_id(report_id),
        }
    )


@app.route("/api/export-training-data", methods=["GET"])
@login_required(role="doctor")
def export_training_data():
    db = get_db()
    rows = db.execute(
        """
        SELECT features_json, clinician_confirmed_label
        FROM reports
        WHERE consented_for_training = 1
          AND clinician_confirmed_label IN ('confirmed_pd', 'confirmed_healthy')
        ORDER BY created_at ASC, id ASC
        """
    ).fetchall()

    records = []
    for row in rows:
        label = 1 if row["clinician_confirmed_label"] == "confirmed_pd" else 0
        records.append(
            {
                "features": json.loads(row["features_json"] or "{}"),
                "label": label,
            }
        )

    response = jsonify(records)
    response.headers["X-Record-Count"] = str(len(records))
    return response


@app.route("/stream-chunk", methods=["POST"])
def stream_chunk():
    if "audio" not in request.files:
        return jsonify({"error": "No audio chunk supplied.", "skip": True}), 200

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No audio chunk supplied.", "skip": True}), 200
    try:
        ensure_supported_audio_file(audio_file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc), "skip": True}), 200
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(audio_file.filename)) as temp:
        audio_file.save(temp.name)
        temp_path = temp.name

    try:
        result = run_prediction(temp_path, gender=normalize_gender(request.form.get("gender")))
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc), "skip": True}), 200
    except Exception as exc:
        app.logger.exception("Chunk analysis failed")
        return jsonify({"error": str(exc), "skip": False}), 200
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@app.route("/analyze-voice", methods=["POST"])
def analyze_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "Please choose an audio file before analyzing."}), 400
    try:
        ensure_supported_audio_file(audio_file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix(audio_file.filename)) as temp:
        audio_file.save(temp.name)
        temp_path = temp.name

    try:
        result = analyze_voice_sample(
            temp_path,
            gender=normalize_gender(request.form.get("gender")),
        )
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Voice analysis failed")
        return jsonify({"error": "Could not analyze the uploaded audio."}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


with app.app_context():
    init_db()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
