from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import tempfile
from feature_extractor import extract_features

app = Flask(__name__)
CORS(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model  = joblib.load(os.path.join(MODELS_DIR, "parkinsons_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

rfe_path  = os.path.join(MODELS_DIR, "rfe_selector.pkl")
rfe_selector = joblib.load(rfe_path) if os.path.exists(rfe_path) else None

ALL_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

def run_prediction(audio_path):
    features_dict  = extract_features(audio_path)
    feature_vector = [features_dict[f] for f in ALL_FEATURES]
    X = np.array(feature_vector).reshape(1, -1)
    X_scaled = scaler.transform(X)
    if rfe_selector is not None:
        X_scaled = X_scaled[:, rfe_selector.support_]
    prediction  = model.predict(X_scaled)
    probability = model.predict_proba(X_scaled)[0].tolist()
    return {
        "prediction":    int(prediction[0]),
        "label":         "Parkinson's Detected" if prediction[0] == 1 else "Healthy",
        "probabilities": {
            "healthy":    round(probability[0], 4),
            "parkinsons": round(probability[1], 4),
        },
        "features": features_dict
    }

@app.route("/")
def home():
    return jsonify({"status": "Live Parkinson Voice API running"})

@app.route("/stream-chunk", methods=["POST"])
def stream_chunk():
    if "audio" not in request.files:
        return jsonify({"error": "No audio chunk", "skip": True}), 200

    audio_file = request.files["audio"]
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = run_prediction(tmp_path)
        print(f"[OK] Prediction: {result['label']} | "
              f"Healthy: {result['probabilities']['healthy']:.2f} | "
              f"PD: {result['probabilities']['parkinsons']:.2f}")
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        # Return the error to the frontend so it shows up
        return jsonify({"error": str(e), "skip": False}), 200
    finally:
        os.unlink(tmp_path)

@app.route("/analyze-voice", methods=["POST"])
def analyze_voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    audio_file = request.files["audio"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name
    try:
        return jsonify(run_prediction(tmp_path))
    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    app.run(debug=True, port=5000)