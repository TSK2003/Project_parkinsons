import json
import os

import joblib
import pandas as pd
from imblearn.over_sampling import KMeansSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons.data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("=" * 64)
print("Parkinson's Disease Training Pipeline")
print("=" * 64)

print("\n[1/6] Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["name"])
X = df.drop(columns=["status"])
y = df["status"]
feature_names = list(X.columns)
print(f"Dataset shape : {X.shape}")
print(f"Class balance : Healthy={sum(y == 0)}, PD={sum(y == 1)}")

print("\n[2/6] Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n[3/6] Building cross-validated search pipeline...")
feature_selector = RFE(
    estimator=LogisticRegression(max_iter=3000, random_state=42),
    n_features_to_select=15,
)

search_pipeline = ImbPipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("smote", KMeansSMOTE(random_state=42, cluster_balance_threshold=0.1)),
        ("selector", feature_selector),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
    ]
)

param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=3000, random_state=42)],
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["lbfgs", "liblinear"],
    },
    {
        "classifier": [SVC(probability=True, random_state=42)],
        "classifier__C": [1, 10, 30],
        "classifier__gamma": ["scale", 0.1, 0.01],
    },
    {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [None, 8, 12],
        "classifier__min_samples_split": [2, 4],
    },
    {
        "classifier": [ExtraTreesClassifier(random_state=42)],
        "classifier__n_estimators": [300, 500],
        "classifier__max_depth": [None, 8, 12],
    },
    {
        "classifier": [GradientBoostingClassifier(random_state=42)],
        "classifier__n_estimators": [100, 200],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__max_depth": [2, 3],
    },
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=search_pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1,
    verbose=0,
    refit=True,
)

print("\n[4/6] Running model search...")
grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_
best_classifier = best_pipeline.named_steps["classifier"]
best_selector = best_pipeline.named_steps["selector"]
best_scaler = best_pipeline.named_steps["scaler"]
selected_mask = best_selector.support_
selected_features = [
    feature_names[index] for index, selected in enumerate(selected_mask) if selected
]

print(f"Best classifier : {type(best_classifier).__name__}")
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")
print("Selected features:")
for feature in selected_features:
    print(f"  - {feature}")

print("\n[5/6] Evaluating on the holdout test set...")
test_predictions = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Test accuracy   : {test_accuracy * 100:.2f}%")
print(classification_report(y_test, test_predictions, target_names=["Healthy", "Parkinson's"]))

cm = confusion_matrix(y_test, test_predictions)
print("Confusion matrix:")
print(f"  True Healthy : {cm[0][0]}  |  False Positive : {cm[0][1]}")
print(f"  False Negative: {cm[1][0]} |  True PD        : {cm[1][1]}")

print("\n[6/6] Saving inference artifacts...")
# Copy the fitted preprocessing stack from the best search result. This keeps
# inference identical to the trained pipeline without applying SMOTE at runtime.
inference_pipeline = Pipeline(
    steps=[
        ("scaler", best_scaler),
        ("selector", best_selector),
        ("classifier", best_classifier),
    ]
)

joblib.dump(inference_pipeline, os.path.join(MODELS_DIR, "prediction_pipeline.pkl"))
joblib.dump(best_classifier, os.path.join(MODELS_DIR, "parkinsons_model.pkl"))
joblib.dump(best_scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(best_selector, os.path.join(MODELS_DIR, "rfe_selector.pkl"))
joblib.dump(selected_features, os.path.join(MODELS_DIR, "selected_features.pkl"))
joblib.dump(selected_mask, os.path.join(MODELS_DIR, "selected_mask.pkl"))

metadata = {
    "best_classifier": type(best_classifier).__name__,
    "best_params": {
        key: str(value) if key == "classifier" else value
        for key, value in grid.best_params_.items()
    },
    "cv_accuracy": round(float(grid.best_score_), 4),
    "test_accuracy": round(float(test_accuracy), 4),
    "selected_features": selected_features,
}

with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

print(f"Saved artifacts to: {MODELS_DIR}")
print("Training complete. Run python3 app.py to start the portal.")
