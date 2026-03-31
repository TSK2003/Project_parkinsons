import json
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons.data")
MERGED_DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons_merged.data")
ALIGNED_MERGED_DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons_merged_aligned.data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR = os.path.join(MODELS_DIR, "evaluation")

RANDOM_STATE = 42
OUTER_SPLITS = 4
INNER_SPLITS = 4
TARGET_PD_RECALL = 0.80
TARGET_BALANCED_ACCURACY = 0.88
THRESHOLD_GRID = np.round(np.linspace(0.25, 0.85, 25), 3)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


ALL_DATASET_FEATURES = [
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
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]

FEATURE_SUBSETS = {
    "paper_4": ["HNR", "RPDE", "DFA", "PPE"],
    "paper_7": [
        "MDVP:Jitter(Abs)",
        "Jitter:DDP",
        "MDVP:APQ",
        "HNR",
        "RPDE",
        "DFA",
        "PPE",
    ],
    "nonlinear_5": ["HNR", "RPDE", "DFA", "D2", "PPE"],
    "robust_8": [
        "MDVP:Jitter(Abs)",
        "MDVP:APQ",
        "HNR",
        "RPDE",
        "DFA",
        "spread1",
        "spread2",
        "PPE",
    ],
    "full_22": ALL_DATASET_FEATURES,
}


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_candidate_configs() -> list[dict]:
    def build_soft_voting_ensemble() -> VotingClassifier:
        return VotingClassifier(
            estimators=[
                (
                    "et",
                    ExtraTreesClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=100,
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ],
            voting="soft",
        )

    return [
        {
            "name": "nonlinear_5_extra_trees",
            "feature_set": "nonlinear_5",
            "features": FEATURE_SUBSETS["nonlinear_5"],
            "model_name": "ExtraTreesClassifier",
            "build_estimator": lambda: build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        },
        {
            "name": "nonlinear_5_svm",
            "feature_set": "nonlinear_5",
            "features": FEATURE_SUBSETS["nonlinear_5"],
            "model_name": "SVC",
            "build_estimator": lambda: build_pipeline(
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    C=10,
                    gamma="scale",
                    random_state=RANDOM_STATE,
                )
            ),
        },
        {
            "name": "nonlinear_5_ensemble",
            "feature_set": "nonlinear_5",
            "features": FEATURE_SUBSETS["nonlinear_5"],
            "model_name": "VotingClassifier",
            "build_estimator": lambda: build_pipeline(build_soft_voting_ensemble()),
        },
        {
            "name": "paper_4_extra_trees",
            "feature_set": "paper_4",
            "features": FEATURE_SUBSETS["paper_4"],
            "model_name": "ExtraTreesClassifier",
            "build_estimator": lambda: build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        },
        {
            "name": "paper_4_logreg",
            "feature_set": "paper_4",
            "features": FEATURE_SUBSETS["paper_4"],
            "model_name": "LogisticRegression",
            "build_estimator": lambda: build_pipeline(
                LogisticRegression(
                    max_iter=5000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        },
        {
            "name": "paper_7_random_forest",
            "feature_set": "paper_7",
            "features": FEATURE_SUBSETS["paper_7"],
            "model_name": "RandomForestClassifier",
            "build_estimator": lambda: build_pipeline(
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=None,
                    min_samples_split=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        },
        {
            "name": "robust_8_extra_trees",
            "feature_set": "robust_8",
            "features": FEATURE_SUBSETS["robust_8"],
            "model_name": "ExtraTreesClassifier",
            "build_estimator": lambda: build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        },
        {
            "name": "robust_8_gradient_boost",
            "feature_set": "robust_8",
            "features": FEATURE_SUBSETS["robust_8"],
            "model_name": "GradientBoostingClassifier",
            "build_estimator": lambda: build_pipeline(
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                )
            ),
        },
        {
            "name": "full_22_gradient_boost",
            "feature_set": "full_22",
            "features": FEATURE_SUBSETS["full_22"],
            "model_name": "GradientBoostingClassifier",
            "build_estimator": lambda: build_pipeline(
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    random_state=RANDOM_STATE,
                )
            ),
        },
        {
            "name": "full_22_extra_trees",
            "feature_set": "full_22",
            "features": FEATURE_SUBSETS["full_22"],
            "model_name": "ExtraTreesClassifier",
            "build_estimator": lambda: build_pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        },
    ]


def validate_feature_subsets(feature_names: list[str]) -> None:
    missing_by_subset = {}
    for subset_name, subset_features in FEATURE_SUBSETS.items():
        missing = [feature for feature in subset_features if feature not in feature_names]
        if missing:
            missing_by_subset[subset_name] = missing

    if missing_by_subset:
        details = "; ".join(
            f"{subset}: {', '.join(features)}"
            for subset, features in missing_by_subset.items()
        )
        raise ValueError(f"Configured feature subsets reference missing dataset columns: {details}")


def extract_subject_ids(names: pd.Series) -> pd.Series:
    def derive_subject_id(name: str) -> str:
        stem = os.path.splitext(str(name))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "phon" and parts[1].startswith("R") and parts[2].startswith("S"):
            return "_".join(parts[:3])
        return stem

    return names.fillna("").map(derive_subject_id)


def aggregate_subject_predictions(prediction_rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(prediction_rows)
    subject_level = (
        frame.groupby("subject_id", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            parkinsons_probability=("parkinsons_probability", "mean"),
            healthy_probability=("healthy_probability", "mean"),
            threshold=("threshold", "first"),
            recording_count=("recording_name", "count"),
            fold=("fold", "first"),
            candidate_name=("candidate_name", "first"),
            feature_set=("feature_set", "first"),
            model_name=("model_name", "first"),
        )
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    subject_level["prediction"] = (
        subject_level["parkinsons_probability"] >= subject_level["threshold"]
    ).astype(int)
    subject_level["label"] = np.where(
        subject_level["prediction"] == 1,
        "Parkinson's Detected",
        "Healthy",
    )
    return subject_level


def compute_subject_metrics(subject_level: pd.DataFrame) -> dict:
    y_true = subject_level["y_true"].to_numpy()
    y_pred = subject_level["prediction"].to_numpy()
    y_score = subject_level["parkinsons_probability"].to_numpy()

    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "pd_recall_sensitivity": float(recall_score(y_true, y_pred, pos_label=1)),
        "healthy_recall_specificity": float(recall_score(y_true, y_pred, pos_label=0)),
        "f1_score": float(f1_score(y_true, y_pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def candidate_rank(metrics: dict) -> tuple:
    return (
        int(metrics["pd_recall_sensitivity"] >= TARGET_PD_RECALL),
        round(metrics["balanced_accuracy"], 6),
        round(metrics["healthy_recall_specificity"], 6),
        round(metrics["pd_recall_sensitivity"], 6),
        round(metrics["f1_score"], 6),
    )


def threshold_rank(metrics: dict) -> tuple:
    return (
        int(metrics["pd_recall_sensitivity"] >= TARGET_PD_RECALL),
        round(metrics["balanced_accuracy"], 6),
        round(metrics["healthy_recall_specificity"], 6),
        round(metrics["pd_recall_sensitivity"], 6),
        round(metrics["f1_score"], 6),
    )


def tune_threshold(subject_probabilities: pd.DataFrame) -> dict:
    best = None
    for threshold in THRESHOLD_GRID:
        candidate_frame = subject_probabilities.copy()
        candidate_frame["threshold"] = float(threshold)
        candidate_frame["prediction"] = (
            candidate_frame["parkinsons_probability"] >= threshold
        ).astype(int)
        metrics = compute_subject_metrics(candidate_frame)
        candidate = {
            "threshold": float(threshold),
            "metrics": metrics,
        }
        if best is None or threshold_rank(metrics) > threshold_rank(best["metrics"]):
            best = candidate
    return best


def fit_estimator_with_smote(
    candidate: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    estimator = candidate["build_estimator"]()
    train_subset = X_train[candidate["features"]]
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(train_subset, y_train)
    estimator.fit(X_train_res, y_train_res)
    return estimator


def evaluate_candidate_with_inner_cv(
    candidate: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> dict:
    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []
    for inner_fold, (inner_train_index, inner_valid_index) in enumerate(
        inner_cv.split(X_train, y_train, groups=groups_train),
        start=1,
    ):
        inner_X_train = X_train.iloc[inner_train_index]
        inner_y_train = y_train.iloc[inner_train_index]
        inner_X_valid = X_train.iloc[inner_valid_index]
        inner_y_valid = y_train.iloc[inner_valid_index]

        estimator = fit_estimator_with_smote(candidate, inner_X_train, inner_y_train)
        probabilities = estimator.predict_proba(
            inner_X_valid[candidate["features"]]
        )[:, 1]

        for subject_id, y_true, pd_probability in zip(
            groups_train.iloc[inner_valid_index],
            inner_y_valid,
            probabilities,
        ):
            rows.append(
                {
                    "subject_id": subject_id,
                    "recording_name": f"inner_fold_{inner_fold}",
                    "y_true": int(y_true),
                    "parkinsons_probability": float(pd_probability),
                    "healthy_probability": float(1.0 - pd_probability),
                    "threshold": 0.5,
                    "fold": inner_fold,
                    "candidate_name": candidate["name"],
                    "feature_set": candidate["feature_set"],
                    "model_name": candidate["model_name"],
                }
            )

    subject_probabilities = aggregate_subject_predictions(rows)
    threshold_choice = tune_threshold(subject_probabilities)

    return {
        "candidate_name": candidate["name"],
        "feature_set": candidate["feature_set"],
        "features": candidate["features"],
        "model_name": candidate["model_name"],
        "threshold": threshold_choice["threshold"],
        "inner_subject_metrics": threshold_choice["metrics"],
    }


def write_curve_csv(
    file_path: str,
    column_a: str,
    values_a: np.ndarray,
    column_b: str,
    values_b: np.ndarray,
) -> None:
    pd.DataFrame({column_a: values_a, column_b: values_b}).to_csv(file_path, index=False)


def write_line_chart_svg(
    file_path: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    baseline: float | None = None,
    diagonal_reference: bool = False,
) -> None:
    width = 720
    height = 520
    left = 80
    right = 30
    top = 50
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    def project(x_val: float, y_val: float) -> tuple[float, float]:
        x_pos = left + (float(x_val) * plot_width)
        y_pos = top + ((1.0 - float(y_val)) * plot_height)
        return x_pos, y_pos

    points = " ".join(
        f"{x_pos:.2f},{y_pos:.2f}"
        for x_pos, y_pos in (project(x, y) for x, y in zip(x_values, y_values))
    )

    tick_lines = []
    tick_labels = []
    for tick in np.linspace(0, 1, 6):
        x_tick, _ = project(tick, 0)
        _, y_tick = project(0, tick)
        tick_lines.append(
            f'<line x1="{x_tick:.2f}" y1="{top}" x2="{x_tick:.2f}" y2="{top + plot_height}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        tick_lines.append(
            f'<line x1="{left}" y1="{y_tick:.2f}" x2="{left + plot_width}" y2="{y_tick:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{x_tick:.2f}" y="{height - 45}" text-anchor="middle" '
            'font-size="12" fill="#475569">{tick:.1f}</text>'
        )
        tick_labels.append(
            f'<text x="{left - 16}" y="{y_tick + 4:.2f}" text-anchor="end" '
            'font-size="12" fill="#475569">{tick:.1f}</text>'
        )

    baseline_line = ""
    if baseline is not None:
        x_start, y_start = project(0.0, baseline)
        x_end, y_end = project(1.0, baseline)
        baseline_line = (
            f'<line x1="{x_start:.2f}" y1="{y_start:.2f}" '
            f'x2="{x_end:.2f}" y2="{y_end:.2f}" '
            'stroke="#f59e0b" stroke-width="2" stroke-dasharray="8 6" />'
        )

    diagonal_line = ""
    if diagonal_reference:
        x_start, y_start = project(0.0, 0.0)
        x_end, y_end = project(1.0, 1.0)
        diagonal_line = (
            f'<line x1="{x_start:.2f}" y1="{y_start:.2f}" '
            f'x2="{x_end:.2f}" y2="{y_end:.2f}" '
            'stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 6" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="22" font-weight="700" fill="#0f172a">{title}</text>
  {''.join(tick_lines)}
  <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#0f172a" stroke-width="2" />
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#0f172a" stroke-width="2" />
  {diagonal_line}
  {baseline_line}
  <polyline fill="none" stroke="#14746f" stroke-width="4" points="{points}" />
  {''.join(tick_labels)}
  <text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" font-size="16" fill="#0f172a">{x_label}</text>
  <text x="24" y="{height / 2:.0f}" text-anchor="middle" font-size="16" fill="#0f172a" transform="rotate(-90 24 {height / 2:.0f})">{y_label}</text>
</svg>
"""

    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def write_confusion_matrix_artifacts(file_stem: str, matrix: np.ndarray) -> None:
    with open(f"{file_stem}.json", "w", encoding="utf-8") as handle:
        json.dump({"labels": ["Healthy", "Parkinson's"], "matrix": matrix.tolist()}, handle, indent=2)

    width = 520
    height = 420
    start_x = 150
    start_y = 120
    cell_size = 110
    flat_max = max(int(matrix.max()), 1)
    labels = ["Healthy", "Parkinson's"]

    cells = []
    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            intensity = 0.15 + (0.75 * (value / flat_max))
            fill = f"rgba(20, 116, 111, {intensity:.3f})"
            x_pos = start_x + (column_index * cell_size)
            y_pos = start_y + (row_index * cell_size)
            cells.append(
                f'<rect x="{x_pos}" y="{y_pos}" width="{cell_size}" height="{cell_size}" '
                f'fill="{fill}" stroke="#0f172a" stroke-width="1.5" />'
            )
            cells.append(
                f'<text x="{x_pos + cell_size / 2:.1f}" y="{y_pos + cell_size / 2 + 8:.1f}" '
                'text-anchor="middle" font-size="28" font-weight="700" fill="#0f172a">'
                f"{value}</text>"
            )

    x_labels = "".join(
        f'<text x="{start_x + (index * cell_size) + (cell_size / 2):.1f}" y="{start_y - 18}" '
        'text-anchor="middle" font-size="16" fill="#0f172a">'
        f"{label}</text>"
        for index, label in enumerate(labels)
    )
    y_labels = "".join(
        f'<text x="{start_x - 18}" y="{start_y + (index * cell_size) + (cell_size / 2) + 6:.1f}" '
        'text-anchor="end" font-size="16" fill="#0f172a">'
        f"{label}</text>"
        for index, label in enumerate(labels)
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="{width}" height="{height}" fill="#ffffff" />
  <text x="{width / 2:.0f}" y="38" text-anchor="middle" font-size="24" font-weight="700" fill="#0f172a">Subject-Level Confusion Matrix</text>
  <text x="{start_x + cell_size:.0f}" y="86" text-anchor="middle" font-size="18" fill="#0f172a">Predicted label</text>
  <text x="44" y="{start_y + cell_size:.0f}" text-anchor="middle" font-size="18" fill="#0f172a" transform="rotate(-90 44 {start_y + cell_size:.0f})">True label</text>
  {x_labels}
  {y_labels}
  {''.join(cells)}
</svg>
"""

    with open(f"{file_stem}.svg", "w", encoding="utf-8") as handle:
        handle.write(svg)


def print_metric_block(title: str, metrics: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Balanced accuracy      : {metrics['balanced_accuracy']:.4f}")
    print(f"PD recall (sensitivity): {metrics['pd_recall_sensitivity']:.4f}")
    print(f"Healthy recall         : {metrics['healthy_recall_specificity']:.4f}")
    print(f"F1 score               : {metrics['f1_score']:.4f}")
    print(f"ROC AUC                : {metrics['roc_auc']:.4f}")
    print(f"Average precision      : {metrics['average_precision']:.4f}")
    print(
        "Goal targets           : "
        f"PD recall {'PASS' if metrics['pd_recall_sensitivity'] >= TARGET_PD_RECALL else 'FAIL'}, "
        f"balanced accuracy {'PASS' if metrics['balanced_accuracy'] >= TARGET_BALANCED_ACCURACY else 'FAIL'}"
    )


def relative_path(path: str) -> str:
    return os.path.relpath(path, BASE_DIR)


def resolve_dataset_path() -> str:
    if os.path.exists(ALIGNED_MERGED_DATA_PATH):
        return ALIGNED_MERGED_DATA_PATH
    if os.path.exists(MERGED_DATA_PATH):
        return MERGED_DATA_PATH
    return DEFAULT_DATA_PATH


def main() -> None:
    print("=" * 72)
    print("Parkinson's Disease Training Pipeline - Subset + Threshold Tuning")
    print("=" * 72)

    print("\n[1/7] Loading dataset and deriving subject groups...")
    dataset_path = resolve_dataset_path()
    print(f"Using dataset          : {dataset_path}")
    df = pd.read_csv(dataset_path)
    df["subject_id"] = extract_subject_ids(df["name"])
    subject_status = df.groupby("subject_id")["status"].first()

    print(f"Recording count        : {len(df)}")
    print(f"Feature count          : {len(df.columns) - 3}")
    print(f"Subject count          : {subject_status.shape[0]}")
    print(f"Healthy subjects       : {int((subject_status == 0).sum())}")
    print(f"Parkinson's subjects   : {int((subject_status == 1).sum())}")

    feature_names = [column for column in df.columns if column not in {"name", "status", "subject_id"}]
    validate_feature_subsets(feature_names)
    X = df[feature_names]
    y = df["status"]
    groups = df["subject_id"]

    candidate_configs = build_candidate_configs()

    print("\n[2/7] Running nested grouped validation with threshold tuning...")
    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    outer_prediction_rows: list[dict] = []
    fold_summaries: list[dict] = []

    for fold_index, (train_index, test_index) in enumerate(
        outer_cv.split(X, y, groups=groups),
        start=1,
    ):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        groups_train = groups.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        groups_test = groups.iloc[test_index]
        names_test = df.iloc[test_index]["name"]

        candidate_results = []
        for candidate in candidate_configs:
            result = evaluate_candidate_with_inner_cv(candidate, X_train, y_train, groups_train)
            candidate_results.append(result)

        best_candidate = max(candidate_results, key=lambda item: candidate_rank(item["inner_subject_metrics"]))
        estimator_template = next(
            candidate for candidate in candidate_configs if candidate["name"] == best_candidate["candidate_name"]
        )
        final_fold_estimator = fit_estimator_with_smote(estimator_template, X_train, y_train)
        test_probabilities = final_fold_estimator.predict_proba(X_test[best_candidate["features"]])[:, 1]

        for recording_name, subject_id, y_true, pd_probability in zip(
            names_test,
            groups_test,
            y_test,
            test_probabilities,
        ):
            outer_prediction_rows.append(
                {
                    "recording_name": recording_name,
                    "subject_id": subject_id,
                    "y_true": int(y_true),
                    "parkinsons_probability": float(pd_probability),
                    "healthy_probability": float(1.0 - pd_probability),
                    "threshold": float(best_candidate["threshold"]),
                    "fold": fold_index,
                    "candidate_name": best_candidate["candidate_name"],
                    "feature_set": best_candidate["feature_set"],
                    "model_name": best_candidate["model_name"],
                }
            )

        fold_subjects = aggregate_subject_predictions(
            [row for row in outer_prediction_rows if row["fold"] == fold_index]
        )
        fold_metrics = compute_subject_metrics(fold_subjects)
        fold_summary = {
            "fold": fold_index,
            "best_candidate": best_candidate["candidate_name"],
            "feature_set": best_candidate["feature_set"],
            "model_name": best_candidate["model_name"],
            "threshold": round(float(best_candidate["threshold"]), 4),
            "inner_subject_metrics": {
                key: round(value, 4)
                for key, value in best_candidate["inner_subject_metrics"].items()
            },
            "outer_subject_metrics": {
                key: round(value, 4)
                for key, value in fold_metrics.items()
            },
        }
        fold_summaries.append(fold_summary)

        print(
            f"  Fold {fold_index}: {best_candidate['candidate_name']} | "
            f"threshold={best_candidate['threshold']:.3f} | "
            f"balanced_accuracy={fold_metrics['balanced_accuracy']:.4f} | "
            f"healthy_recall={fold_metrics['healthy_recall_specificity']:.4f} | "
            f"pd_recall={fold_metrics['pd_recall_sensitivity']:.4f}"
        )

    print("\n[3/7] Aggregating outer-fold predictions at subject level...")
    recording_eval_path = os.path.join(EVAL_DIR, "recording_level_oof_predictions.csv")
    subject_eval_path = os.path.join(EVAL_DIR, "subject_level_oof_predictions.csv")
    recording_eval = pd.DataFrame(outer_prediction_rows).sort_values(
        ["fold", "subject_id", "recording_name"]
    )
    recording_eval.to_csv(recording_eval_path, index=False)

    subject_eval = aggregate_subject_predictions(outer_prediction_rows)
    subject_eval.to_csv(subject_eval_path, index=False)

    subject_metrics = compute_subject_metrics(subject_eval)
    print_metric_block("Subject-Level Out-of-Fold Metrics", subject_metrics)

    print("\n[4/7] Writing evaluation artifacts...")
    subject_y_true = subject_eval["y_true"].to_numpy()
    subject_y_pred = subject_eval["prediction"].to_numpy()
    subject_y_score = subject_eval["parkinsons_probability"].to_numpy()

    subject_confusion_matrix_stem = os.path.join(EVAL_DIR, "subject_confusion_matrix")
    roc_curve_csv_path = os.path.join(EVAL_DIR, "roc_curve.csv")
    precision_recall_curve_csv_path = os.path.join(EVAL_DIR, "precision_recall_curve.csv")
    roc_curve_svg_path = os.path.join(EVAL_DIR, "roc_curve.svg")
    precision_recall_curve_svg_path = os.path.join(EVAL_DIR, "precision_recall_curve.svg")

    subject_cm = confusion_matrix(subject_y_true, subject_y_pred, labels=[0, 1])
    write_confusion_matrix_artifacts(subject_confusion_matrix_stem, subject_cm)

    fpr, tpr, _ = roc_curve(subject_y_true, subject_y_score)
    precision, recall, _ = precision_recall_curve(subject_y_true, subject_y_score)
    write_curve_csv(roc_curve_csv_path, "fpr", fpr, "tpr", tpr)
    write_curve_csv(
        precision_recall_curve_csv_path,
        "recall",
        recall,
        "precision",
        precision,
    )
    write_line_chart_svg(
        roc_curve_svg_path,
        x_values=fpr,
        y_values=tpr,
        title="Subject-Level ROC Curve",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        diagonal_reference=True,
    )
    write_line_chart_svg(
        precision_recall_curve_svg_path,
        x_values=recall,
        y_values=precision,
        title="Subject-Level Precision-Recall Curve",
        x_label="Recall",
        y_label="Precision",
        baseline=float(subject_y_true.mean()),
    )

    print("\n[5/7] Choosing the final model and threshold on all subjects...")
    final_candidate_results = []
    for candidate in candidate_configs:
        final_candidate_results.append(
            evaluate_candidate_with_inner_cv(candidate, X, y, groups)
        )

    final_choice = max(
        final_candidate_results,
        key=lambda item: candidate_rank(item["inner_subject_metrics"]),
    )
    final_config = next(
        candidate for candidate in candidate_configs if candidate["name"] == final_choice["candidate_name"]
    )
    final_estimator = fit_estimator_with_smote(final_config, X, y)

    print(f"Best candidate          : {final_choice['candidate_name']}")
    print(f"Feature set             : {final_choice['feature_set']}")
    print(f"Model                   : {final_choice['model_name']}")
    print(f"Decision threshold      : {final_choice['threshold']:.3f}")
    print("Selected features:")
    for feature in final_choice["features"]:
        print(f"  - {feature}")

    print("\n[6/7] Saving inference artifacts...")
    prediction_pipeline_path = os.path.join(MODELS_DIR, "prediction_pipeline.pkl")
    selected_features_path = os.path.join(MODELS_DIR, "selected_features.pkl")
    model_metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")

    joblib.dump(final_estimator, prediction_pipeline_path)
    joblib.dump(final_choice["features"], selected_features_path)

    metadata = {
        "training_mode": "subject_grouped_subset_threshold_tuning_with_smote",
        "scoring_metric": "balanced_accuracy_with_pd_recall_floor",
        "sklearn_version": sklearn.__version__,
        "dataset_path_used": relative_path(dataset_path),
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "decision_threshold": round(float(final_choice["threshold"]), 4),
        "target_pd_recall": TARGET_PD_RECALL,
        "target_balanced_accuracy": TARGET_BALANCED_ACCURACY,
        "best_candidate": final_choice["candidate_name"],
        "feature_set_name": final_choice["feature_set"],
        "best_classifier": final_choice["model_name"],
        "selected_features": final_choice["features"],
        "artifact_paths": {
            "prediction_pipeline": relative_path(prediction_pipeline_path),
            "selected_features": relative_path(selected_features_path),
            "model_metadata": relative_path(model_metadata_path),
        },
        "subject_level_oof_metrics": {
            key: round(value, 4)
            for key, value in subject_metrics.items()
        },
        "final_candidate_inner_metrics": {
            key: round(value, 4)
            for key, value in final_choice["inner_subject_metrics"].items()
        },
        "evaluation_artifacts": {
            "recording_level_oof_predictions": relative_path(recording_eval_path),
            "subject_level_oof_predictions": relative_path(subject_eval_path),
            "subject_confusion_matrix_json": relative_path(f"{subject_confusion_matrix_stem}.json"),
            "subject_confusion_matrix_svg": relative_path(f"{subject_confusion_matrix_stem}.svg"),
            "roc_curve_csv": relative_path(roc_curve_csv_path),
            "roc_curve_svg": relative_path(roc_curve_svg_path),
            "precision_recall_curve_csv": relative_path(precision_recall_curve_csv_path),
            "precision_recall_curve_svg": relative_path(precision_recall_curve_svg_path),
        },
        "fold_summaries": fold_summaries,
        "final_candidate_results": [
            {
                "candidate_name": item["candidate_name"],
                "feature_set": item["feature_set"],
                "model_name": item["model_name"],
                "threshold": round(float(item["threshold"]), 4),
                "inner_subject_metrics": {
                    key: round(value, 4)
                    for key, value in item["inner_subject_metrics"].items()
                },
            }
            for item in final_candidate_results
        ],
    }

    with open(model_metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\n[7/7] Saved final model, subset, and tuned threshold.")
    print(f"Model artifacts         : {MODELS_DIR}")
    print(f"Evaluation artifacts    : {EVAL_DIR}")


if __name__ == "__main__":
    main()
