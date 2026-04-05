from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
UCI_DATASET_PATH = DATA_DIR / "parkinsons.data"
HC_DIR = DATA_DIR / "HC_AH"
PD_DIR = DATA_DIR / "PD_AH"
MERGE_SCRIPT_PATH = ROOT_DIR / "merge_datasets.py"
TRAIN_SCRIPT_PATH = ROOT_DIR / "train_model.py"
FEATURE_EXTRACTOR_PATH = ROOT_DIR / "feature_extractor.py"
MIN_EXPECTED_FEATURES = 22


@dataclass
class CheckResult:
    section: str
    name: str
    passed: bool
    details: str


RESULTS: list[CheckResult] = []


def add_result(section: str, name: str, passed: bool, details: str) -> None:
    RESULTS.append(CheckResult(section=section, name=name, passed=passed, details=details))


def print_divider(title: str) -> None:
    print("")
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_check(result: CheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status:<4}] {result.name}: {result.details}")


def find_wav_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(path for path in folder.rglob("*.wav") if path.is_file())


def check_path_exists(section: str, name: str, path: Path, *, expect_dir: bool = False) -> bool:
    exists = path.is_dir() if expect_dir else path.is_file()
    kind = "directory" if expect_dir else "file"
    details = f"{kind} found at {path}" if exists else f"{kind} missing at {path}"
    add_result(section, name, exists, details)
    return exists


def get_package_version(distribution_name: str) -> tuple[bool, str]:
    try:
        version = importlib.metadata.version(distribution_name)
        return True, version
    except importlib.metadata.PackageNotFoundError:
        return False, "not installed"
    except Exception as exc:
        return False, f"version lookup failed: {exc}"


def load_feature_extractor_module():
    spec = importlib.util.spec_from_file_location("project_feature_extractor", FEATURE_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {FEATURE_EXTRACTOR_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_extractor_callable(module) -> tuple[Callable[[str], dict] | None, str, bool]:
    extractor_class = getattr(module, "FeatureExtractor", None)
    if extractor_class is not None:
        try:
            instance = extractor_class()
            if hasattr(instance, "extract_features"):
                return instance.extract_features, "FeatureExtractor().extract_features", True
            return None, "FeatureExtractor class exists but has no extract_features() method", False
        except Exception as exc:
            return None, f"FeatureExtractor class exists but could not be instantiated: {exc}", False

    extractor_function = getattr(module, "extract_features", None)
    if callable(extractor_function):
        return extractor_function, "module-level extract_features()", True

    return None, "No FeatureExtractor class or extract_features() function was found", False


def run_file_checks() -> tuple[list[Path], list[Path]]:
    print_divider("1. FILE CHECKS")

    check_path_exists("FILE CHECKS", "UCI dataset", UCI_DATASET_PATH)
    hc_exists = check_path_exists("FILE CHECKS", "Healthy WAV folder", HC_DIR, expect_dir=True)
    pd_exists = check_path_exists("FILE CHECKS", "Parkinson's WAV folder", PD_DIR, expect_dir=True)
    check_path_exists("FILE CHECKS", "merge_datasets.py", MERGE_SCRIPT_PATH)
    check_path_exists("FILE CHECKS", "train_model.py", TRAIN_SCRIPT_PATH)
    check_path_exists("FILE CHECKS", "feature_extractor.py", FEATURE_EXTRACTOR_PATH)

    hc_wavs = find_wav_files(HC_DIR) if hc_exists else []
    pd_wavs = find_wav_files(PD_DIR) if pd_exists else []

    hc_ok = len(hc_wavs) >= 1
    pd_ok = len(pd_wavs) >= 1
    add_result(
        "FILE CHECKS",
        "Healthy WAV count",
        hc_ok,
        f"{len(hc_wavs)} WAV file(s) found in {HC_DIR}",
    )
    add_result(
        "FILE CHECKS",
        "Parkinson's WAV count",
        pd_ok,
        f"{len(pd_wavs)} WAV file(s) found in {PD_DIR}",
    )

    for result in [item for item in RESULTS if item.section == "FILE CHECKS"]:
        print_check(result)

    return hc_wavs, pd_wavs


def run_package_checks() -> None:
    print_divider("2. PACKAGE CHECKS")

    package_names = [
        ("scikit-learn", "scikit-learn"),
        ("imbalanced-learn", "imbalanced-learn"),
        ("librosa", "librosa"),
        ("praat-parselmouth", "praat-parselmouth"),
        ("nolds", "nolds"),
    ]

    for label, distribution_name in package_names:
        installed, version = get_package_version(distribution_name)
        details = f"version {version}" if installed else version
        add_result("PACKAGE CHECKS", label, installed, details)

    for result in [item for item in RESULTS if item.section == "PACKAGE CHECKS"]:
        print_check(result)


def run_feature_extractor_check(pd_wavs: list[Path]) -> None:
    print_divider("3. FEATURE EXTRACTOR CHECK")

    if not FEATURE_EXTRACTOR_PATH.is_file():
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Feature extractor module import",
            False,
            f"{FEATURE_EXTRACTOR_PATH} does not exist",
        )
        print_check(RESULTS[-1])
        return

    try:
        module = load_feature_extractor_module()
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "feature_extractor.py import",
            True,
            f"imported successfully from {FEATURE_EXTRACTOR_PATH}",
        )
    except Exception as exc:
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "feature_extractor.py import",
            False,
            f"import failed: {exc}",
        )
        print_check(RESULTS[-1])
        return

    extractor_callable, extractor_source, extractor_ok = resolve_extractor_callable(module)
    add_result(
        "FEATURE EXTRACTOR CHECK",
        "Feature extractor interface",
        extractor_ok,
        extractor_source,
    )

    if not pd_wavs:
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Sample WAV selection",
            False,
            f"No WAV files were found in {PD_DIR}",
        )
        for result in [item for item in RESULTS if item.section == "FEATURE EXTRACTOR CHECK"]:
            print_check(result)
        return

    sample_wav = pd_wavs[0]
    add_result(
        "FEATURE EXTRACTOR CHECK",
        "Sample WAV selection",
        True,
        f"Using {sample_wav.name}",
    )

    feature_payload = None
    extraction_error = None
    if extractor_callable is not None:
        try:
            feature_payload = extractor_callable(str(sample_wav))
        except Exception as exc:
            extraction_error = exc
    else:
        extraction_error = RuntimeError("No callable extractor was available")

    extraction_passed = isinstance(feature_payload, dict) and len(feature_payload) > 0
    if extraction_error is not None:
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Feature extraction run",
            False,
            f"{type(extraction_error).__name__}: {extraction_error}",
        )
    elif feature_payload is None:
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Feature extraction run",
            False,
            "extract_features() returned None without an exception",
        )
    elif not isinstance(feature_payload, dict):
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Feature extraction run",
            False,
            f"extract_features() returned {type(feature_payload).__name__}, expected dict",
        )
    else:
        add_result(
            "FEATURE EXTRACTOR CHECK",
            "Feature extraction run",
            extraction_passed,
            f"returned {len(feature_payload)} feature(s)",
        )

    feature_count = len(feature_payload) if isinstance(feature_payload, dict) else 0
    add_result(
        "FEATURE EXTRACTOR CHECK",
        "Feature count threshold",
        feature_count >= MIN_EXPECTED_FEATURES,
        f"{feature_count} feature(s) returned; expected at least {MIN_EXPECTED_FEATURES}",
    )

    for result in [item for item in RESULTS if item.section == "FEATURE EXTRACTOR CHECK"]:
        print_check(result)

    if isinstance(feature_payload, dict):
        print("")
        print(f"Feature values for {sample_wav.name}")
        print("-" * 72)
        for feature_name in sorted(feature_payload):
            print(f"{feature_name}: {feature_payload[feature_name]}")
        print("-" * 72)
        print(f"Total feature count: {len(feature_payload)}")


def run_dataset_balance_check(hc_wavs: list[Path], pd_wavs: list[Path]) -> None:
    print_divider("4. DATASET BALANCE CHECK")

    dataset_loaded = False
    dataset = None
    try:
        dataset = pd.read_csv(UCI_DATASET_PATH)
        dataset_loaded = True
        add_result(
            "DATASET BALANCE CHECK",
            "Load UCI dataset",
            True,
            f"loaded {UCI_DATASET_PATH.name} successfully",
        )
    except Exception as exc:
        add_result(
            "DATASET BALANCE CHECK",
            "Load UCI dataset",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    total_rows = pd_rows = healthy_rows = 0
    if dataset_loaded and dataset is not None:
        if "status" not in dataset.columns:
            add_result(
                "DATASET BALANCE CHECK",
                "UCI dataset status column",
                False,
                "status column is missing",
            )
        else:
            total_rows = int(len(dataset))
            pd_rows = int((dataset["status"] == 1).sum())
            healthy_rows = int((dataset["status"] == 0).sum())
            class_ratio = (
                f"{pd_rows}:{healthy_rows}"
                if healthy_rows > 0
                else f"{pd_rows}:0"
            )
            add_result(
                "DATASET BALANCE CHECK",
                "UCI class counts",
                total_rows > 0 and pd_rows > 0 and healthy_rows > 0,
                f"total rows={total_rows}, PD rows={pd_rows}, Healthy rows={healthy_rows}, ratio={class_ratio}",
            )

    total_new_wavs = len(hc_wavs) + len(pd_wavs)
    estimated_successful_rows = math.floor(total_new_wavs * 0.80)
    estimated_total_rows_after_merge = total_rows + estimated_successful_rows
    estimated_pd_added = math.floor(len(pd_wavs) * 0.80)
    estimated_hc_added = math.floor(len(hc_wavs) * 0.80)

    add_result(
        "DATASET BALANCE CHECK",
        "Healthy WAV inventory",
        len(hc_wavs) > 0,
        f"{len(hc_wavs)} healthy WAV file(s) found",
    )
    add_result(
        "DATASET BALANCE CHECK",
        "Parkinson's WAV inventory",
        len(pd_wavs) > 0,
        f"{len(pd_wavs)} Parkinson's WAV file(s) found",
    )
    add_result(
        "DATASET BALANCE CHECK",
        "Merged row estimate",
        dataset_loaded and total_new_wavs > 0,
        (
            f"estimated merged rows={estimated_total_rows_after_merge} "
            f"(original={total_rows} + estimated successful WAV rows={estimated_successful_rows}; "
            f"estimated added PD={estimated_pd_added}, added Healthy={estimated_hc_added})"
        ),
    )

    for result in [item for item in RESULTS if item.section == "DATASET BALANCE CHECK"]:
        print_check(result)


def print_summary() -> None:
    print_divider("5. SUMMARY")

    sections = [
        "FILE CHECKS",
        "PACKAGE CHECKS",
        "FEATURE EXTRACTOR CHECK",
        "DATASET BALANCE CHECK",
    ]

    overall_ready = True
    for section in sections:
        section_results = [result for result in RESULTS if result.section == section]
        section_passed = all(result.passed for result in section_results) if section_results else False
        overall_ready = overall_ready and section_passed
        status = "PASS" if section_passed else "FAIL"
        passed_count = sum(result.passed for result in section_results)
        total_count = len(section_results)
        print(f"[{status:<4}] {section}: {passed_count}/{total_count} checks passed")

    print("")
    print("Detailed failures")
    print("-" * 72)
    failures = [result for result in RESULTS if not result.passed]
    if failures:
        for result in failures:
            print(f"- {result.section} -> {result.name}: {result.details}")
    else:
        print("None")

    print("")
    print(
        "Ready to merge: YES"
        if overall_ready
        else "Ready to merge: NO — fix items above"
    )


def main() -> int:
    print("=" * 72)
    print("Parkinson's Dataset Merge Verification")
    print("=" * 72)
    print(f"Project root: {ROOT_DIR}")

    hc_wavs, pd_wavs = run_file_checks()
    run_package_checks()
    run_feature_extractor_check(pd_wavs)
    run_dataset_balance_check(hc_wavs, pd_wavs)
    print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
