from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from feature_extractor import extract_features


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UCI_DATASET_PATH = DATA_DIR / "parkinsons.data"
MERGED_DATASET_PATH = DATA_DIR / "parkinsons_merged.data"
ALIGNED_MERGED_DATASET_PATH = DATA_DIR / "parkinsons_merged_aligned.data"
SUMMARY_PATH = DATA_DIR / "parkinsons_merged_summary.json"
SKIPPED_CSV_PATH = DATA_DIR / "parkinsons_merged_skipped.csv"
CALIBRATION_STATS_PATH = DATA_DIR / "parkinsons_alignment_stats.json"
HC_WAV_DIR = DATA_DIR / "HC_AH"
PD_WAV_DIR = DATA_DIR / "PD_AH"


@dataclass
class SkipRecord:
    folder_label: str
    filename: str
    reason: str


def print_divider(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_found(label: str, path: Path) -> None:
    print(f"[OK] FOUND  {label}: {path}")


def print_missing(label: str, path: Path) -> None:
    print(f"[ERROR] MISSING {label}: {path}")


def load_base_dataset() -> pd.DataFrame:
    dataset = pd.read_csv(UCI_DATASET_PATH)
    if "name" not in dataset.columns or "status" not in dataset.columns:
        raise ValueError("The base UCI dataset must contain 'name' and 'status' columns.")
    return dataset


def build_record_name(folder_label: str, wav_path: Path) -> str:
    safe_stem = wav_path.stem.replace(" ", "_")
    return f"figshare_{folder_label.lower()}_{safe_stem}"


def find_wav_files(folder: Path) -> list[Path]:
    return sorted(path for path in folder.rglob("*.wav") if path.is_file())


def safe_extract_features(wav_path: Path) -> dict:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"scipy\.signal\._signaltools",
        )
        return extract_features(str(wav_path))


def process_wav_folder(
    folder: Path,
    folder_label: str,
    status_value: int,
    feature_columns: list[str],
) -> tuple[list[dict], list[SkipRecord]]:
    wav_paths = find_wav_files(folder)
    rows: list[dict] = []
    skipped: list[SkipRecord] = []

    print_divider(f"Processing {folder_label} WAV Files ({len(wav_paths)})")

    for index, wav_path in enumerate(wav_paths, start=1):
        prefix = f"[{index:>3}/{len(wav_paths)}]"

        try:
            extracted = safe_extract_features(wav_path)
            missing_features = [name for name in feature_columns if name not in extracted]
            if missing_features:
                raise KeyError(
                    "missing extracted features: " + ", ".join(missing_features)
                )

            row = {
                "name": build_record_name(folder_label, wav_path),
                **{name: extracted[name] for name in feature_columns},
                "status": status_value,
            }
            rows.append(row)
            print(f"{prefix} [OK]  {wav_path.name}")
        except ValueError as exc:
            reason = f"quality_fail: {exc}"
            skipped.append(
                SkipRecord(folder_label=folder_label, filename=wav_path.name, reason=reason)
            )
            print(f"{prefix} [SKIP] {wav_path.name} - {reason}")
        except Exception as exc:
            reason = f"error({type(exc).__name__}: {exc})"
            skipped.append(
                SkipRecord(folder_label=folder_label, filename=wav_path.name, reason=reason)
            )
            print(f"{prefix} [SKIP] {wav_path.name} - {reason}")

    return rows, skipped


def write_summary(
    base_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    aligned_merged_df: pd.DataFrame,
    pd_rows: list[dict],
    hc_rows: list[dict],
    skipped: list[SkipRecord],
    calibration_stats: list[dict],
) -> None:
    summary = {
        "base_dataset_path": str(UCI_DATASET_PATH),
        "merged_dataset_path": str(MERGED_DATASET_PATH),
        "aligned_merged_dataset_path": str(ALIGNED_MERGED_DATASET_PATH),
        "base_row_count": int(len(base_df)),
        "base_pd_rows": int((base_df["status"] == 1).sum()),
        "base_healthy_rows": int((base_df["status"] == 0).sum()),
        "added_pd_rows": len(pd_rows),
        "added_healthy_rows": len(hc_rows),
        "merged_row_count": int(len(merged_df)),
        "merged_pd_rows": int((merged_df["status"] == 1).sum()),
        "merged_healthy_rows": int((merged_df["status"] == 0).sum()),
        "aligned_merged_row_count": int(len(aligned_merged_df)),
        "aligned_merged_pd_rows": int((aligned_merged_df["status"] == 1).sum()),
        "aligned_merged_healthy_rows": int((aligned_merged_df["status"] == 0).sum()),
        "calibration_stats_path": str(CALIBRATION_STATS_PATH),
        "calibrated_feature_count": len(calibration_stats),
        "skipped_file_count": len(skipped),
        "skipped_files": [asdict(item) for item in skipped],
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with CALIBRATION_STATS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(calibration_stats, handle, indent=2)

    skipped_frame = pd.DataFrame([asdict(item) for item in skipped])
    skipped_frame.to_csv(SKIPPED_CSV_PATH, index=False)


def align_source_to_reference(
    source_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    aligned_df = source_df.copy()
    calibration_stats: list[dict] = []

    for feature_name in feature_columns:
        source_mean = float(source_df[feature_name].mean())
        source_std = float(source_df[feature_name].std())
        reference_mean = float(reference_df[feature_name].mean())
        reference_std = float(reference_df[feature_name].std())

        if source_std <= 1e-12 or reference_std <= 1e-12:
            calibration_stats.append(
                {
                    "feature": feature_name,
                    "applied": False,
                    "reason": "source_or_reference_std_too_small",
                    "source_mean": source_mean,
                    "source_std": source_std,
                    "reference_mean": reference_mean,
                    "reference_std": reference_std,
                }
            )
            continue

        aligned_df[feature_name] = (
            (source_df[feature_name] - source_mean) / source_std
        ) * reference_std + reference_mean

        calibration_stats.append(
            {
                "feature": feature_name,
                "applied": True,
                "source_mean": source_mean,
                "source_std": source_std,
                "reference_mean": reference_mean,
                "reference_std": reference_std,
                "mean_shift": reference_mean - source_mean,
            }
        )

    return aligned_df, calibration_stats


def main() -> int:
    print("=" * 72)
    print("Parkinson's Dataset Merge")
    print("=" * 72)

    missing_inputs = False
    if UCI_DATASET_PATH.is_file():
        print_found("UCI dataset", UCI_DATASET_PATH)
    else:
        print_missing("UCI dataset", UCI_DATASET_PATH)
        missing_inputs = True

    if PD_WAV_DIR.is_dir():
        print_found("PD WAV folder", PD_WAV_DIR)
    else:
        print_missing("PD WAV folder", PD_WAV_DIR)
        missing_inputs = True

    if HC_WAV_DIR.is_dir():
        print_found("HC WAV folder", HC_WAV_DIR)
    else:
        print_missing("HC WAV folder", HC_WAV_DIR)
        missing_inputs = True

    if missing_inputs:
        print("")
        print("MERGE ABORTED")
        return 1

    print("[OK] Feature extractor loaded successfully")

    base_df = load_base_dataset()
    column_order = list(base_df.columns)
    feature_columns = [column for column in column_order if column not in {"name", "status"}]

    print("")
    print(
        f"Base dataset rows      : {len(base_df)} "
        f"(PD={(base_df['status'] == 1).sum()}, Healthy={(base_df['status'] == 0).sum()})"
    )
    print(f"Compatible feature count: {len(feature_columns)}")

    pd_rows, pd_skipped = process_wav_folder(
        folder=PD_WAV_DIR,
        folder_label="PD",
        status_value=1,
        feature_columns=feature_columns,
    )
    hc_rows, hc_skipped = process_wav_folder(
        folder=HC_WAV_DIR,
        folder_label="HC",
        status_value=0,
        feature_columns=feature_columns,
    )

    extracted_rows = pd_rows + hc_rows
    skipped = pd_skipped + hc_skipped

    extracted_df = pd.DataFrame(extracted_rows)
    if extracted_df.empty:
        print("")
        print("No new WAV rows were extracted successfully. Nothing to merge.")
        return 1

    extracted_df = extracted_df[column_order]
    merged_df = pd.concat([base_df, extracted_df], ignore_index=True)
    aligned_base_df, calibration_stats = align_source_to_reference(
        source_df=base_df,
        reference_df=extracted_df,
        feature_columns=feature_columns,
    )
    aligned_merged_df = pd.concat(
        [aligned_base_df[column_order], extracted_df],
        ignore_index=True,
    )
    merged_df.to_csv(MERGED_DATASET_PATH, index=False)
    aligned_merged_df.to_csv(ALIGNED_MERGED_DATASET_PATH, index=False)
    write_summary(
        base_df,
        merged_df,
        aligned_merged_df,
        pd_rows,
        hc_rows,
        skipped,
        calibration_stats,
    )

    print_divider("Merge Summary")
    print(f"Original rows          : {len(base_df)}")
    print(f"Added PD rows          : {len(pd_rows)} / {len(find_wav_files(PD_WAV_DIR))}")
    print(f"Added Healthy rows     : {len(hc_rows)} / {len(find_wav_files(HC_WAV_DIR))}")
    print(f"Skipped files          : {len(skipped)}")
    print(f"Merged total rows      : {len(merged_df)}")
    print(f"Merged PD rows         : {(merged_df['status'] == 1).sum()}")
    print(f"Merged Healthy rows    : {(merged_df['status'] == 0).sum()}")
    print(f"Merged dataset path    : {MERGED_DATASET_PATH}")
    print(f"Aligned dataset path   : {ALIGNED_MERGED_DATASET_PATH}")
    print(f"Calibrated features    : {sum(item['applied'] for item in calibration_stats)}")
    print(f"Calibration stats path : {CALIBRATION_STATS_PATH}")
    print(f"Summary JSON path      : {SUMMARY_PATH}")
    print(f"Skipped CSV path       : {SKIPPED_CSV_PATH}")

    if skipped:
        print("")
        print("Skipped files")
        print("-" * 72)
        for item in skipped:
            print(f"- {item.folder_label}: {item.filename} - {item.reason}")

    print("")
    print("MERGE COMPLETE")

    if not pd_rows or not hc_rows:
        print("WARNING: One folder produced zero extracted rows. Check skipped file reasons above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
