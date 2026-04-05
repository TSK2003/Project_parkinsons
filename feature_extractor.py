import librosa
import nolds
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.linalg import toeplitz
from scipy.signal import resample_poly, wiener

MIN_DURATION_SECONDS = 2.0
MIN_RMS = 0.002
MIN_VOICED_SECONDS = 0.7
MIN_VOICED_FRAMES = 20
MAX_CLIPPED_FRACTION = 0.01
WIENER_VARIANCE_FLOOR = 1e-12

TARGET_SAMPLE_RATE = 44100
PITCH_STRENGTH_FLOOR = 0.18
PITCH_FRAME_STEP_SECONDS = 0.01
MAX_UNVOICED_GAP_FRAMES = 10

PERIOD_FLOOR = 0.0001
PERIOD_CEILING = 0.02
MAX_PERIOD_FACTOR = 1.3
MAX_AMPLITUDE_FACTOR = 1.6

LINEAR_PREDICTION_ORDER = 4
MAX_NONLINEAR_SAMPLES = 1500
C3_HZ = 130.81278265

TRADITIONAL_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
]

NONLINEAR_FEATURE_NAMES = [
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

MFCC_FEATURE_NAMES = [
    key
    for index in range(1, 14)
    for key in (
        f"MFCC_{index}_mean",
        f"MFCC_{index}_std",
        f"MFCC_{index}_delta_mean",
    )
]

ALL_FEATURE_NAMES = TRADITIONAL_FEATURE_NAMES + NONLINEAR_FEATURE_NAMES + MFCC_FEATURE_NAMES


def _get_pitch_range(gender=None):
    if gender == "male":
        return 75, 300
    if gender == "female":
        return 150, 500
    return 65, 500


def extract_features(audio_path: str, gender: str | None = None) -> dict:
    sound = parselmouth.Sound(audio_path)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()

    raw_waveform = np.asarray(sound.values).reshape(-1).astype(np.float64)
    duration = float(sound.duration)
    if duration < MIN_DURATION_SECONDS:
        raise ValueError(
            "The voice sample is too short. Please record at least 8 to 10 seconds of a steady 'Ahh'."
        )

    raw_rms = float(np.sqrt(np.mean(np.square(raw_waveform)))) if raw_waveform.size else 0.0
    if raw_rms < MIN_RMS:
        raise ValueError(
            "The voice sample is too quiet. Please speak louder and move closer to the microphone."
        )

    clipped_fraction = float(np.mean(np.abs(raw_waveform) >= 0.98)) if raw_waveform.size else 0.0
    if clipped_fraction > MAX_CLIPPED_FRACTION:
        raise ValueError(
            "The voice sample is clipped or distorted. Please reduce microphone gain and record again."
        )

    raw_waveform = _denoise_waveform(raw_waveform)
    sound = parselmouth.Sound(raw_waveform, sound.sampling_frequency)
    prepared_sound = _normalize_and_resample(sound)
    voiced_sound, voiced_pitch = _extract_longest_voiced_region(prepared_sound, gender=gender)

    traditional_features = _extract_traditional_features(voiced_sound, gender=gender)
    nonlinear_features = _extract_nonlinear_features(voiced_sound, voiced_pitch)
    mfcc_features = _extract_mfcc_features(
        np.asarray(voiced_sound.values).reshape(-1),
        int(round(voiced_sound.sampling_frequency)),
    )

    return traditional_features | nonlinear_features | mfcc_features


def _denoise_waveform(waveform: np.ndarray) -> np.ndarray:
    cleaned_waveform = np.nan_to_num(
        np.asarray(waveform, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if (
        cleaned_waveform.size < 3
        or float(np.var(cleaned_waveform)) <= WIENER_VARIANCE_FLOOR
    ):
        return cleaned_waveform.astype(np.float32)

    return np.nan_to_num(
        wiener(cleaned_waveform),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def _normalize_and_resample(sound: parselmouth.Sound) -> parselmouth.Sound:
    waveform = np.asarray(sound.values).reshape(-1).astype(np.float64)
    sample_rate = int(round(sound.sampling_frequency))

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = resample_poly(waveform, TARGET_SAMPLE_RATE, sample_rate)
        sample_rate = TARGET_SAMPLE_RATE

    waveform = waveform - float(np.mean(waveform))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 0:
        waveform = 0.95 * waveform / peak

    return parselmouth.Sound(waveform, sample_rate)


def _extract_longest_voiced_region(
    sound: parselmouth.Sound,
    gender: str | None = None,
) -> tuple[parselmouth.Sound, parselmouth.Pitch]:
    pitch_floor, pitch_ceiling = _get_pitch_range(gender)
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    pitch_array = pitch.selected_array
    frequencies = np.asarray(pitch_array["frequency"], dtype=np.float64)
    strengths = np.asarray(pitch_array["strength"], dtype=np.float64)
    times = np.asarray(pitch.xs(), dtype=np.float64)

    voiced_mask = (
        np.isfinite(frequencies)
        & (frequencies > 0)
        & np.isfinite(strengths)
        & (strengths >= PITCH_STRENGTH_FLOOR)
    )
    voiced_mask = _fill_short_false_runs(voiced_mask, MAX_UNVOICED_GAP_FRAMES)
    if not np.any(voiced_mask):
        raise ValueError(
            "Not enough voiced frames were detected. Please record in a quiet room and sustain a steady vowel."
        )

    start_index, end_index = _longest_true_run(voiced_mask)
    if start_index is None or end_index is None:
        raise ValueError(
            "Not enough voiced frames were detected. Please record in a quiet room and sustain a steady vowel."
        )

    frame_step = (
        float(np.median(np.diff(times))) if len(times) > 1 else PITCH_FRAME_STEP_SECONDS
    )
    voiced_duration = (end_index - start_index + 1) * frame_step
    if voiced_duration < MIN_VOICED_SECONDS:
        raise ValueError(
            "The app detected sound, but not enough steady single-pitch 'Ahh'. Please hold one continuous 'Ahh' a little longer without changing pitch."
        )

    margin = frame_step * 2
    start_time = max(float(times[start_index]) - margin, 0.0)
    end_time = min(float(times[end_index]) + margin, float(sound.duration))
    voiced_sound = sound.extract_part(start_time, end_time, preserve_times=False)
    voiced_pitch = call(voiced_sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    return voiced_sound, voiced_pitch


def _longest_true_run(mask: np.ndarray) -> tuple[int | None, int | None]:
    best_start = None
    best_end = None
    current_start = None

    for index, flag in enumerate(mask):
        if flag and current_start is None:
            current_start = index
        elif not flag and current_start is not None:
            current_end = index - 1
            if best_start is None or current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            current_start = None

    if current_start is not None:
        current_end = len(mask) - 1
        if best_start is None or current_end - current_start > best_end - best_start:
            best_start, best_end = current_start, current_end

    return best_start, best_end


def _fill_short_false_runs(mask: np.ndarray, max_gap: int) -> np.ndarray:
    bridged_mask = np.asarray(mask, dtype=bool).copy()
    if bridged_mask.size == 0 or max_gap <= 0:
        return bridged_mask

    gap_start = None
    for index, flag in enumerate(bridged_mask):
        if not flag and gap_start is None:
            gap_start = index
        elif flag and gap_start is not None:
            gap_length = index - gap_start
            has_voiced_left = gap_start > 0 and bridged_mask[gap_start - 1]
            has_voiced_right = index < bridged_mask.size and bridged_mask[index]
            if has_voiced_left and has_voiced_right and gap_length <= max_gap:
                bridged_mask[gap_start:index] = True
            gap_start = None

    return bridged_mask


def _extract_traditional_features(
    sound: parselmouth.Sound,
    gender: str | None = None,
) -> dict:
    pitch_floor, pitch_ceiling = _get_pitch_range(gender)
    pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)

    fo = _safe(call(pitch, "Get mean", 0, 0, "Hertz"), None)
    fhi = _safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"), None)
    flo = _safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"), None)

    jitter_pct = _measure_point_process(point_process, "Get jitter (local)")
    jitter_abs = _measure_point_process(point_process, "Get jitter (local, absolute)")
    rap = _measure_point_process(point_process, "Get jitter (rap)")
    ppq = _measure_point_process(point_process, "Get jitter (ppq5)")
    ddp = _measure_point_process(point_process, "Get jitter (ddp)")
    if ddp is None and rap is not None:
        ddp = rap * 3.0

    sh = _measure_shimmer(sound, point_process, "Get shimmer (local)")
    sh_db = _measure_shimmer(sound, point_process, "Get shimmer (local_dB)")
    apq3 = _measure_shimmer(sound, point_process, "Get shimmer (apq3)")
    apq5 = _measure_shimmer(sound, point_process, "Get shimmer (apq5)")
    apq = _measure_shimmer(sound, point_process, "Get shimmer (apq11)")
    dda = _measure_shimmer(sound, point_process, "Get shimmer (dda)")
    if dda is None and apq3 is not None:
        dda = apq3 * 3.0

    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
        hnr = _safe(call(harmonicity, "Get mean", 0, 0), None)
    except Exception:
        hnr = None

    nhr = None
    if hnr is not None:
        nhr = float(1.0 / (10 ** (hnr / 10.0))) if hnr > 0 else None

    critical_values = {
        "MDVP:Fo(Hz)": fo,
        "MDVP:Fhi(Hz)": fhi,
        "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_pct,
        "MDVP:Jitter(Abs)": jitter_abs,
        "MDVP:RAP": rap,
        "MDVP:PPQ": ppq,
        "Jitter:DDP": ddp,
        "MDVP:Shimmer": sh,
        "MDVP:Shimmer(dB)": sh_db,
        "Shimmer:APQ3": apq3,
        "Shimmer:APQ5": apq5,
        "MDVP:APQ": apq,
        "Shimmer:DDA": dda,
        "NHR": nhr,
        "HNR": hnr,
    }
    missing = [name for name, value in critical_values.items() if value is None]
    if missing:
        raise ValueError(
            "The recording quality is not stable enough for perturbation analysis. Please record a cleaner steady vowel."
        )

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
    }


def _extract_mfcc_features(audio_array, sr):
    """
    Extract 13 MFCC coefficients (mean + std each + delta mean).
    MFCCs capture vocal tract shape and complement the classic
    Parkinson's perturbation and nonlinear biomarkers.
    """
    audio_array = np.asarray(audio_array, dtype=np.float32)
    sample_rate = int(sr)

    mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    result = {}
    for index in range(13):
        result[f"MFCC_{index + 1}_mean"] = float(np.mean(mfccs[index]))
        result[f"MFCC_{index + 1}_std"] = float(np.std(mfccs[index]))
        result[f"MFCC_{index + 1}_delta_mean"] = float(np.mean(delta_mfccs[index]))
    return result


def _extract_nonlinear_features(
    sound: parselmouth.Sound,
    pitch: parselmouth.Pitch,
) -> dict:
    waveform = np.asarray(sound.values).reshape(-1).astype(np.float64)
    signal_series = _downsample_series(waveform, MAX_NONLINEAR_SAMPLES)
    if signal_series.size < 200:
        raise ValueError("The voiced segment is too short for nonlinear feature extraction.")

    rpde = _safe(_rpde(signal_series), None)
    dfa_val = _safe(nolds.dfa(signal_series), None)
    try:
        d2 = _safe(nolds.corr_dim(signal_series, emb_dim=2), None)
    except Exception:
        d2 = None

    pitch_array = pitch.selected_array
    pitch_values = np.asarray(pitch_array["frequency"], dtype=np.float64)
    pitch_strengths = np.asarray(pitch_array["strength"], dtype=np.float64)
    voiced_pitch_values = pitch_values[
        np.isfinite(pitch_values)
        & (pitch_values > 0)
        & np.isfinite(pitch_strengths)
        & (pitch_strengths >= PITCH_STRENGTH_FLOOR)
    ]

    if voiced_pitch_values.size < MIN_VOICED_FRAMES:
        raise ValueError(
            "Not enough voiced frames were detected. Please record in a quiet room and sustain a steady vowel."
        )

    spread1, spread2, ppe = _pitch_distribution_features(voiced_pitch_values)
    nonlinear_values = {
        "RPDE": rpde,
        "DFA": dfa_val,
        "spread1": spread1,
        "spread2": spread2,
        "D2": d2,
        "PPE": ppe,
    }
    missing = [name for name, value in nonlinear_values.items() if value is None]
    if missing:
        raise ValueError(
            "The recording does not contain enough stable pitch detail for nonlinear feature extraction."
        )

    return {
        "RPDE": round(rpde, 5),
        "DFA": round(dfa_val, 5),
        "spread1": round(spread1, 5),
        "spread2": round(spread2, 5),
        "D2": round(d2, 5),
        "PPE": round(ppe, 5),
    }


def _measure_point_process(point_process, command_name: str) -> float | None:
    try:
        return _safe(
            call(
                point_process,
                command_name,
                0,
                0,
                PERIOD_FLOOR,
                PERIOD_CEILING,
                MAX_PERIOD_FACTOR,
            ),
            None,
        )
    except Exception:
        return None


def _measure_shimmer(sound, point_process, command_name: str) -> float | None:
    try:
        return _safe(
            call(
                [sound, point_process],
                command_name,
                0,
                0,
                PERIOD_FLOOR,
                PERIOD_CEILING,
                MAX_PERIOD_FACTOR,
                MAX_AMPLITUDE_FACTOR,
            ),
            None,
        )
    except Exception:
        return None


def _downsample_series(values: np.ndarray, max_length: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    if series.size <= max_length:
        return series - float(np.mean(series))

    step = int(np.ceil(series.size / max_length))
    series = series[::step]
    return series - float(np.mean(series))


def _pitch_distribution_features(f0_series: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if len(f0_series) < MIN_VOICED_FRAMES:
        return None, None, None

    semitone_pitch = 12.0 * np.log2(f0_series / C3_HZ)
    residual = _whiten_sequence(semitone_pitch, LINEAR_PREDICTION_ORDER)
    residual = residual[np.isfinite(residual)]
    if residual.size < max(LINEAR_PREDICTION_ORDER + 4, 16):
        return None, None, None

    centered_residual = residual - float(np.median(residual))
    spread1 = float(np.log(np.mean(np.abs(centered_residual)) + 1e-6))
    spread2 = float(np.std(centered_residual))

    histogram, _ = np.histogram(
        centered_residual,
        bins=min(30, max(10, centered_residual.size // 4)),
        density=False,
    )
    histogram = histogram.astype(np.float64)
    histogram = histogram[histogram > 0]
    if histogram.size == 0:
        return None, None, None

    probabilities = histogram / np.sum(histogram)
    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
    ppe = entropy / np.log2(len(probabilities)) if len(probabilities) > 1 else 0.0
    return spread1, spread2, ppe


def _whiten_sequence(values: np.ndarray, order: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float64)
    series = series - float(np.mean(series))
    if series.size <= order + 1:
        return series

    autocorr = np.correlate(series, series, mode="full")[series.size - 1 : series.size + order]
    if autocorr.size < order + 1 or autocorr[0] <= 1e-10:
        return np.diff(series)

    try:
        ar_coefficients = np.linalg.solve(
            toeplitz(autocorr[:-1]),
            autocorr[1:],
        )
    except np.linalg.LinAlgError:
        return np.diff(series)

    residuals = []
    for index in range(order, series.size):
        previous_values = series[index - order : index][::-1]
        predicted = float(np.dot(ar_coefficients, previous_values))
        residuals.append(series[index] - predicted)

    return np.asarray(residuals, dtype=np.float64)


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
    if embedding_count < 2 or epsilon <= 0:
        return None

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
        return None

    counts = np.bincount(periods, minlength=t_max + 1)[1 : t_max + 1].astype(float)
    counts /= counts.sum() + 1e-10
    entropy = -np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))
    return float(entropy / np.log2(t_max)) if t_max > 1 else None
