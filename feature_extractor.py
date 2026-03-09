import numpy as np
import parselmouth
from parselmouth.praat import call
import nolds


def extract_features(audio_path: str) -> dict:
    sound = parselmouth.Sound(audio_path)
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()

    duration = sound.duration
    print(f"  [FEAT] Duration: {duration:.2f}s, SR: {sound.sampling_frequency}Hz")

    # ── Pitch ──────────────────────────────────────────────
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    fo  = _safe(call(pitch, "Get mean",    0, 0, "Hertz"), 120.0)
    fhi = _safe(call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic"), 150.0)
    flo = _safe(call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic"), 90.0)

    # ── Point Process ──────────────────────────────────────
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)

    # ── Jitter ─────────────────────────────────────────────
    def jitter(name):
        try: return _safe(call(point_process, name, 0, 0, 0.0001, 0.02, 1.3), 0.005)
        except: return 0.005

    jitter_pct = jitter("Get jitter (local)")
    jitter_abs = _safe(jitter_pct * 0.00001, 0.00005)
    rap        = jitter("Get jitter (rap)")
    ppq        = jitter("Get jitter (ppq5)")
    ddp        = rap * 3

    # ── Shimmer — wrapped individually so one failure doesn't kill all ───────
    def shimmer(name):
        try: return _safe(call([sound, point_process], name, 0, 0, 0.0001, 0.02, 1.3, 1.6), None)
        except: return None

    sh      = shimmer("Get shimmer (local)")     or 0.04
    sh_db   = shimmer("Get shimmer (local, dB)") or 0.35
    apq3    = shimmer("Get shimmer (apq3)")      or 0.02
    apq5    = shimmer("Get shimmer (apq5)")      or 0.025
    apq     = shimmer("Get shimmer (apq11)")     or 0.03
    dda     = apq3 * 3

    # ── HNR / NHR ──────────────────────────────────────────
    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = _safe(call(harmonicity, "Get mean", 0, 0), 20.0)
    except:
        hnr = 20.0
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.02

    # ── Nonlinear features ─────────────────────────────────
    pitch_values = np.array([
        call(pitch, "Get value at time", t, "Hertz", "Linear")
        for t in np.arange(0, duration, 0.01)
    ])
    pitch_values = pitch_values[~np.isnan(pitch_values)]
    pitch_values = pitch_values[pitch_values > 0]
    print(f"  [FEAT] Voiced frames: {len(pitch_values)}")

    if len(pitch_values) < 10:
        print("  [WARN] Short audio — using fallback nonlinear values")
        rpde, dfa_val, spread1, spread2, d2, ppe = 0.41, 0.82, -4.8, 0.27, 2.3, 0.28
    else:
        if len(pitch_values) < 30:
            pitch_values = np.tile(pitch_values, 3)[:30]
        rpde    = _safe(_rpde(pitch_values), 0.41)
        dfa_val = _safe(nolds.dfa(pitch_values), 0.82)
        log_p   = np.log(pitch_values + 1e-10)
        spread1 = float(np.min(log_p) - np.mean(log_p))
        spread2 = float(np.std(log_p))
        try:    d2 = _safe(nolds.corr_dim(pitch_values, emb_dim=2), 2.3)
        except: d2 = 2.3
        ppe = _safe(_ppe(pitch_values), 0.28)

    print(f"  [OK] fo={fo:.1f} jitter={jitter_pct:.5f} shimmer={sh:.5f} HNR={hnr:.2f}")

    return {
        "MDVP:Fo(Hz)":      round(fo,          5),
        "MDVP:Fhi(Hz)":     round(fhi,         5),
        "MDVP:Flo(Hz)":     round(flo,         5),
        "MDVP:Jitter(%)":   round(jitter_pct,  5),
        "MDVP:Jitter(Abs)": round(jitter_abs,  5),
        "MDVP:RAP":         round(rap,          5),
        "MDVP:PPQ":         round(ppq,          5),
        "Jitter:DDP":       round(ddp,          5),
        "MDVP:Shimmer":     round(sh,           5),
        "MDVP:Shimmer(dB)": round(sh_db,        5),
        "Shimmer:APQ3":     round(apq3,         5),
        "Shimmer:APQ5":     round(apq5,         5),
        "MDVP:APQ":         round(apq,          5),
        "Shimmer:DDA":      round(dda,          5),
        "NHR":              round(nhr,          5),
        "HNR":              round(hnr,          5),
        "RPDE":             round(rpde,         5),
        "DFA":              round(dfa_val,      5),
        "spread1":          round(spread1,      5),
        "spread2":          round(spread2,      5),
        "D2":               round(d2,           5),
        "PPE":              round(ppe,          5),
    }


def _safe(val, fallback):
    try:
        if val is None or np.isnan(val) or np.isinf(val):
            return fallback
        return float(val)
    except:
        return fallback


def _rpde(x, m=4, tau=1, epsilon=None, T_max=None):
    N = len(x)
    if epsilon is None: epsilon = 0.2 * np.std(x)
    if T_max is None:   T_max = min(N // 2, 50)
    M = N - (m - 1) * tau
    if M < 2: return 0.41
    embedded = np.array([x[i:i + m * tau:tau] for i in range(M)])
    periods = []
    for i in range(min(M, 100)):
        dist = np.max(np.abs(embedded - embedded[i]), axis=1)
        close = np.where((dist < epsilon) & (np.arange(M) != i))[0]
        for j in close:
            p = abs(int(j) - int(i))
            if 1 <= p <= T_max: periods.append(p)
    if not periods: return 0.41
    counts = np.bincount(periods, minlength=T_max + 1)[1:T_max + 1].astype(float)
    counts /= counts.sum() + 1e-10
    entropy = -np.sum(counts[counts > 0] * np.log2(counts[counts > 0]))
    return float(entropy / np.log2(T_max)) if T_max > 1 else 0.41


def _ppe(f0_series):
    if len(f0_series) < 2: return 0.28
    semitones = 12 * np.log2(f0_series / (np.median(f0_series) + 1e-10) + 1e-10)
    hist, _ = np.histogram(semitones, bins=min(30, len(f0_series)//2 + 1), density=True)
    hist = hist[hist > 0]
    if len(hist) == 0: return 0.28
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist + 1e-10)))