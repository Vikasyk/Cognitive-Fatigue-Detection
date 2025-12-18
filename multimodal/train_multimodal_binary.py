import os
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "wesad_multimodal_features.csv"

RESULTS_DIR = "results_multimodal_binary"
CONF_MATRIX_PNG = os.path.join(RESULTS_DIR, "confusion_multimodal_binary.png")
METRICS_JSON = os.path.join(RESULTS_DIR, "multimodal_binary_results.json")


def load_data():
    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        raise ValueError("label column missing in CSV.")

    feature_cols = [c for c in df.columns if c not in ["subject_id", "label"]]

    X = df[feature_cols].values
    y_orig = df["label"].astype(int).values

    # Binary labels: non-stress (0,2) vs stress (1)
    y = np.where(y_orig == 1, 1, 0)

    return X, y, feature_cols


def clean_features(X):
    X = np.where(np.isinf(X), np.nan, X)
    X = np.where(np.abs(X) > 1e9, np.nan, X)
    X_df = pd.DataFrame(X)
    X_df = X_df.apply(lambda col: col.fillna(col.median()), axis=0)
    return X_df.values


def show_class_distribution(y, title):
    print(title)
    print(pd.Series(y).value_counts().sort_index())


def tune_random_forest(X_train, y_train):
    X_train = clean_features(X_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        rf,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train_scaled, y_train)

    print("Best params (multimodal, binary):", grid.best_params_)
    print("Best CV macro F1 (multimodal, binary):", grid.best_score_)

    best_rf = grid.best_estimator_
    return best_rf, scaler, grid.best_params_, grid.best_score_


def evaluate_and_plot(model, scaler, X_train, y_train, X_test, y_test):
    X_train = clean_features(X_train)
    X_test = clean_features(X_test)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Test accuracy (multimodal, binary):", acc)
    print("Test macro F1 (multimodal, binary):", f1_macro)
    print("Classification report (multimodal, binary):")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-stress", "Stress"],
        yticklabels=["Non-stress", "Stress"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Multimodal RF (Binary)")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PNG, dpi=200)
    plt.close()

    return acc, f1_macro, report, cm


def save_metrics(best_params, best_cv_f1, acc, f1_macro, report, cm):
    results = {
        "best_params": best_params,
        "best_cv_macro_f1": best_cv_f1,
        "test_accuracy": acc,
        "test_macro_f1": f1_macro,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    with open(METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved metrics to {METRICS_JSON}")


def main():
    X, y, feature_cols = load_data()
    print("Loaded multimodal data with shape:", X.shape)
    print("Feature columns:", feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    X_train = clean_features(X_train)
    X_test = clean_features(X_test)

    show_class_distribution(y_train, "Train class distribution (binary, multimodal):")

    best_rf, scaler, best_params, best_cv_f1 = tune_random_forest(X_train, y_train)

    acc, f1_macro, report, cm = evaluate_and_plot(
        best_rf, scaler, X_train, y_train, X_test, y_test
    )

    save_metrics(best_params, best_cv_f1, acc, f1_macro, report, cm)


if __name__ == "__main__":
    main()
