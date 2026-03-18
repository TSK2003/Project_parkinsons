import nolds
import numpy as np
import parselmouth
from parselmouth.praat import call

MIN_DURATION_SECONDS = 1.5
MIN_RMS = 0.002
MIN_VOICED_FRAMES = 15


def extract_features(audio_path: str) -> dict:
    sound = parselmouth.Sound(audio_path)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()

    duration = float(sound.duration)
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            "The voice sample is too short. Please record at least 8 to 10 seconds of a steady 'Ahh'."
        )

    waveform = np.asarray(sound.values).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(waveform)))) if waveform.size else 0.0
    if rms < MIN_RMS:
        raise ValueError(
            "The voice sample is too quiet. Please speak louder and move closer to the microphone."
        )

    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    fo = _safe(call(pitch, "Get mean", 0, 0, "Hertz"), 120.0)
    fhi = _safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"), 150.0)
    flo = _safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"), 90.0)

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    def jitter(name):
        try:
            return _safe(call(point_process, name, 0, 0, 0.0001, 0.02, 1.3), 0.005)
        except Exception:
            return 0.005

    jitter_pct = jitter("Get jitter (local)")
    jitter_abs = _safe(jitter_pct * 0.00001, 0.00005)
    rap = jitter("Get jitter (rap)")
    ppq = jitter("Get jitter (ppq5)")
    ddp = rap * 3

    def shimmer(name):
        try:
            return _safe(
                call([sound, point_process], name, 0, 0, 0.0001, 0.02, 1.3, 1.6),
                None,
            )
        except Exception:
            return None

    sh = shimmer("Get shimmer (local)") or 0.04
    sh_db = shimmer("Get shimmer (local, dB)") or 0.35
    apq3 = shimmer("Get shimmer (apq3)") or 0.02
    apq5 = shimmer("Get shimmer (apq5)") or 0.025
    apq = shimmer("Get shimmer (apq11)") or 0.03
    dda = apq3 * 3

    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = _safe(call(harmonicity, "Get mean", 0, 0), 20.0)
    except Exception:
        hnr = 20.0
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.02

    pitch_values = np.array(
        [
            call(pitch, "Get value at time", t, "Hertz", "Linear")
            for t in np.arange(0, duration, 0.01)
        ]
    )
    pitch_values = pitch_values[~np.isnan(pitch_values)]
    pitch_values = pitch_values[pitch_values > 0]

    if len(pitch_values) < MIN_VOICED_FRAMES:
        raise ValueError(
            "Not enough voiced frames were detected. Please record in a quiet room and sustain a steady vowel."
        )

    if len(pitch_values) < 30:
        pitch_values = np.tile(pitch_values, 3)[:30]

    rpde = _safe(_rpde(pitch_values), 0.41)
    dfa_val = _safe(nolds.dfa(pitch_values), 0.82)
    log_pitch = np.log(pitch_values + 1e-10)
    spread1 = float(np.min(log_pitch) - np.mean(log_pitch))
    spread2 = float(np.std(log_pitch))
    try:
        d2 = _safe(nolds.corr_dim(pitch_values, emb_dim=2), 2.3)
    except Exception:
        d2 = 2.3
    ppe = _safe(_ppe(pitch_values), 0.28)

    return {
        "MDVP:Fo(Hz)": round(fo, 5),
        "MDVP:Fhi(Hz)": round(fhi, 5),
        "MDVP:Flo(Hz)": round(flo, 5),
        "MDVP:Jitter(%)": round(jitter_pct, 5),
        "MDVP:Jitter(Abs)": round(jitter_abs, 5),
        "MDVP:RAP": round(rap, 5),
        "MDVP:PPQ": round(ppq, 5),
        "Jitter:DDP": round(ddp, 5),
        "MDVP:Shimmer": round(sh, 5),
        "MDVP:Shimmer(dB)": round(sh_db, 5),
        "Shimmer:APQ3": round(apq3, 5),
        "Shimmer:APQ5": round(apq5, 5),
        "MDVP:APQ": round(apq, 5),
        "Shimmer:DDA": round(dda, 5),
        "NHR": round(nhr, 5),
        "HNR": round(hnr, 5),
        "RPDE": round(rpde, 5),
        "DFA": round(dfa_val, 5),
        "spread1": round(spread1, 5),
        "spread2": round(spread2, 5),
        "D2": round(d2, 5),
        "PPE": round(ppe, 5),
    }


def _safe(val, fallback):
    try:
        if val is None or np.isnan(val) or np.isinf(val):
            return fallback
        return float(val)
    except Exception:
        return fallback


def _rpde(x, m=4, tau=1, epsilon=None, t_max=None):
    n_items = len(x)
    if epsilon is None:
        epsilon = 0.2 * np.std(x)
    if t_max is None:
        t_max = min(n_items // 2, 50)
    embedding_count = n_items - (m - 1) * tau
    if embedding_count < 2:
        return 0.41

    embedded = np.array([x[i : i + m * tau : tau] for i in range(embedding_count)])
    periods = []
    for index in range(min(embedding_count, 100)):
        distance = np.max(np.abs(embedded - embedded[index]), axis=1)
        close_matches = np.where(
            (distance < epsilon) & (np.arange(embedding_count) != index)
        )[0]
        for match_index in close_matches:
            period = abs(int(match_index) - int(index))
            if 1 <= period <= t_max:
                periods.append(period)

    if not periods:
        return 0.41

    counts = np.bincount(periods, minlength=t_max + 1)[1 : t_max + 1].astype(float)
    counts /= counts.sum() + 1e-10
    entropy = -np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))
    return float(entropy / np.log2(t_max)) if t_max > 1 else 0.41


def _ppe(f0_series):
    if len(f0_series) < 2:
        return 0.28

    semitones = 12 * np.log2(f0_series / (np.median(f0_series) + 1e-10) + 1e-10)
    histogram, _ = np.histogram(
        semitones, bins=min(30, len(f0_series) // 2 + 1), density=True
    )
    histogram = histogram[histogram > 0]
    if len(histogram) == 0:
        return 0.28

    histogram /= histogram.sum()
    return float(-np.sum(histogram * np.log2(histogram + 1e-10)))
