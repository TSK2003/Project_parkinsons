import argparse
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

SNR_DB = 30.0
PITCH_SHIFT_STEPS = 1
STRETCH_RATES = {
    "stretch_slow": 0.95,
    "stretch_fast": 1.05,
}
VARIANT_SUFFIXES = (
    "pitch_up",
    "pitch_down",
    "stretch_slow",
    "stretch_fast",
    "noise",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate augmented WAV variants for Parkinson's voice training data."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing source WAV files.")
    parser.add_argument("--output-dir", required=True, help="Directory where augmented WAV files will be written.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned augmentations without writing any files.",
    )
    return parser.parse_args()


def discover_wavs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("*.wav") if path.is_file())


def add_gaussian_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.square(audio))) if audio.size else 0.0
    if signal_power <= 0:
        return audio.copy()

    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return audio + noise.astype(np.float32)


def build_variants(audio: np.ndarray, sr: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "pitch_up": librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=PITCH_SHIFT_STEPS),
        "pitch_down": librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=-PITCH_SHIFT_STEPS),
        "stretch_slow": librosa.effects.time_stretch(y=audio, rate=STRETCH_RATES["stretch_slow"]),
        "stretch_fast": librosa.effects.time_stretch(y=audio, rate=STRETCH_RATES["stretch_fast"]),
        "noise": add_gaussian_noise(audio, SNR_DB, rng),
    }


def ensure_parent(path: Path, dry_run: bool) -> None:
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, audio: np.ndarray, sr: int, dry_run: bool) -> None:
    ensure_parent(path, dry_run)
    if dry_run:
        print(f"[dry-run] {path}")
        return

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = np.int16(clipped * 32767)
    wavfile.write(path, sr, pcm)
    print(f"[write] {path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    wav_paths = discover_wavs(input_dir)
    if not wav_paths:
        print("No .wav files were found. Nothing to augment.")
        return

    rng = np.random.default_rng(42)
    augmented_count = 0

    for wav_path in wav_paths:
        audio, sr = librosa.load(wav_path, sr=None, mono=True)
        audio = np.asarray(audio, dtype=np.float32)
        relative_parent = wav_path.relative_to(input_dir).parent
        base_name = wav_path.stem

        variants = build_variants(audio, int(sr), rng)
        for suffix in VARIANT_SUFFIXES:
            output_path = output_dir / relative_parent / f"{base_name}_{suffix}.wav"
            write_wav(output_path, variants[suffix], int(sr), args.dry_run)
            augmented_count += 1

    total_files = len(wav_paths) + augmented_count
    print("")
    print("Augmentation summary")
    print("--------------------")
    print(f"Original count : {len(wav_paths)}")
    print(f"Augmented count: {augmented_count}")
    print(f"Total files    : {total_files}")
    if args.dry_run:
        print("Dry run only   : no files were written")


if __name__ == "__main__":
    main()
