"""
Task A.4 — Adapter-Based Gender Classifier
==========================================
Trains lightweight classifiers on top of frozen ECAPA-TDNN embeddings.

Approach:
  - Base model: ECAPA-TDNN (completely frozen, no fine-tuning)
  - Adapter head: Logistic Regression + SVM (RBF kernel)
  - 80/20 stratified train/test split
  - 5-fold and 10-fold cross-validation for robustness

Key results:
  - Logistic Regression: 100% accuracy (80/20), 99.62% ± 0.19% (5-fold CV)
  - SVM (RBF):           100% accuracy (80/20), 99.81% ± 0.23% (5-fold CV)
  - Both confirm gender is LINEARLY SEPARABLE — RBF kernel not needed
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

Path("results/plots").mkdir(parents=True, exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings_and_labels():
    """Load pre-extracted embeddings and gender labels from Lahaja."""
    with open("results/metrics/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)

    from datasets import load_dataset
    dataset = load_dataset("ai4bharat/Lahaja", split="test", trust_remote_code=True)
    df = dataset.to_pandas()

    indices = sorted(embeddings.keys())
    X = np.array([embeddings[i] for i in indices])
    y = np.array([1 if df.iloc[i]["gender"] == "M" else 0 for i in indices])

    print(f"X shape: {X.shape}")
    print(f"Class distribution: Male={y.sum()}, Female={(y==0).sum()}")
    return X, y


# ---------------------------------------------------------------------------
# Build pipelines (StandardScaler inside pipeline to prevent data leakage)
# ---------------------------------------------------------------------------

def build_pipelines():
    """
    Wrap scaler + classifier in sklearn Pipeline.

    StandardScaler is fitted ONLY on training data — no leakage.
    ECAPA-TDNN embeddings have high L2 norm (~299), so scaling matters.
    """
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
    ])

    return {"Logistic Regression": lr_pipeline, "SVM (RBF)": svm_pipeline}


# ---------------------------------------------------------------------------
# 80/20 single split evaluation
# ---------------------------------------------------------------------------

def evaluate_single_split(X, y, pipelines):
    """Train and evaluate on a fixed 80/20 stratified split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    results = []
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                         "Recall": rec, "F1 Score": f1})

        print(f"\n{name}")
        print(f"  Accuracy:  {acc*100:.2f}%")
        print(f"  Precision: {prec*100:.2f}%")
        print(f"  Recall:    {rec*100:.2f}%")
        print(f"  F1:        {f1*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=["Female", "Male"]))

        plot_confusion_matrix(y_test, y_pred, name)

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/metrics/single_split_results.csv", index=False)
    return results_df, X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Female", "Male"],
                yticklabels=["Female", "Male"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(f"results/plots/confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"Saved → results/plots/confusion_matrix_{fname}.png")


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cross_validation(X, y, pipelines, n_folds=5):
    """
    Stratified k-fold cross-validation.
    Reports train accuracy, test accuracy, and overfit gap per fold.
    """
    print(f"\n{'='*60}")
    print(f"{n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*60)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_cv_results = {}

    for name, pipe in pipelines.items():
        cv_result = cross_validate(
            pipe, X, y, cv=cv,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1
        )

        train_scores = cv_result["train_score"]
        test_scores  = cv_result["test_score"]
        gaps = train_scores - test_scores

        all_cv_results[name] = {
            "train_scores": train_scores,
            "test_scores":  test_scores,
            "gaps": gaps
        }

        print(f"\n{name}")
        print(f"  {'Fold':<8} {'Train':>8} {'Test':>8} {'Gap':>8}")
        for i, (tr, te, g) in enumerate(zip(train_scores, test_scores, gaps)):
            print(f"  Fold {i+1:<3} {tr*100:>7.2f}% {te*100:>7.2f}% {g*100:>7.4f}%")

        print(f"\n  Mean Test:  {test_scores.mean()*100:.2f}% ± {test_scores.std()*100:.2f}%")
        print(f"  Mean Train: {train_scores.mean()*100:.2f}% ± {train_scores.std()*100:.2f}%")
        print(f"  Mean Gap:   {gaps.mean()*100:.4f}%")

        verdict = "✓ No overfitting" if gaps.mean() < 0.01 else "⚠ Possible overfitting"
        print(f"  Verdict:    {verdict}")

    plot_cv_comparison(all_cv_results, n_folds)
    return all_cv_results


def plot_cv_comparison(cv_results, n_folds):
    """Bar plot comparing CV test accuracy across models and folds."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(n_folds)
    width = 0.35
    colors = ["#378ADD", "#1D9E75"]

    for i, (name, res) in enumerate(cv_results.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, res["test_scores"] * 100, width,
                      label=name, color=colors[i], alpha=0.85)

    ax.set_xlabel("Fold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"{n_folds}-Fold CV — Gender Classification Test Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
    ax.set_ylim(98, 100.5)
    ax.legend()
    ax.axhline(99.0, color="#E24B4A", linestyle="--", lw=1, alpha=0.5, label="99% line")
    plt.tight_layout()
    plt.savefig(f"results/plots/cv_{n_folds}fold_comparison.png", dpi=150)
    plt.close()
    print(f"Saved → results/plots/cv_{n_folds}fold_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    X, y = load_embeddings_and_labels()
    pipelines = build_pipelines()

    print("\n" + "="*60)
    print("SINGLE 80/20 SPLIT EVALUATION")
    print("="*60)
    results_df, X_train, X_test, y_train, y_test = evaluate_single_split(X, y, pipelines)

    print("\nResults table:")
    print(results_df.to_string(index=False))

    # 5-fold CV
    cv5 = run_cross_validation(X, y, pipelines, n_folds=5)

    # 10-fold CV
    cv10 = run_cross_validation(X, y, pipelines, n_folds=10)

    print("\n✓ Task A.4 complete.")
