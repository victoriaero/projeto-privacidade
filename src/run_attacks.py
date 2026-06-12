from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "pan15_eng_sentence-transformers__all-MiniLM-L6-v2_mean.npz"
TRAIN_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
TASKS = {
    "gender": ("y_train_gender", "y_test_gender"),
    "age": ("y_train_age", "y_test_age"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitive-attribute inference attacks on cached author embeddings.")
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 34, 55, 89])
    parser.add_argument("--train-fractions", type=float, nargs="+", default=TRAIN_FRACTIONS, help="Fractions of the official train split available to the attacker.")
    return parser.parse_args()


def make_classifiers(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(steps=[("scaler", StandardScaler()), ("classifier", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed))]),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, l2_regularization=1e-3, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="mlogloss", tree_method="hist", random_state=seed, n_jobs=-1),
        "mlp": Pipeline(steps=[("scaler", StandardScaler()), ("classifier", MLPClassifier(hidden_layer_sizes=(64,), activation="relu", alpha=1e-3, max_iter=1000, early_stopping=True, validation_fraction=0.2, n_iter_no_change=20, random_state=seed))]),
    }


def stratified_indices(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"Train fraction must be in (0, 1], got {fraction}")

    rng = np.random.default_rng(seed)

    if fraction == 1:
        indices = np.arange(len(y))
        rng.shuffle(indices)
        return indices

    selected = []
    for label in sorted(np.unique(y)):
        label_indices = np.flatnonzero(y == label)
        n_selected = int(round(len(label_indices) * fraction))
        min_selected = 2 if len(label_indices) >= 2 else 1
        n_selected = max(min_selected, min(n_selected, len(label_indices)))
        selected.extend(rng.choice(label_indices, size=n_selected, replace=False))

    selected = np.array(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    for average in ("macro", "micro", "weighted"):
        metrics[f"precision_{average}"] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f"recall_{average}"] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics[f"f1_{average}"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    return metrics


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def report_rows(report: dict[str, dict[str, float]], context: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict):
            continue
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append({
                **context,
                "class_label": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1-score"],
                "support": values["support"],
            })
    return rows


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, context: dict[str, object]) -> list[dict[str, object]]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            rows.append({
                    **context,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[true_index, pred_index]),
                })
    return rows


def prediction_rows(author_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, context: dict[str, object]) -> list[dict[str, object]]:
    return [{
            **context,
            "author_id": author_id,
            "true_label": true_label,
            "predicted_label": predicted_label,
        }
        for author_id, true_label, predicted_label in zip(author_ids, y_true, y_pred)
    ]


def clear_previous_outputs(output_dir: Path) -> None:
    for filename in ["summary_metrics.csv", "per_class_metrics.csv", "confusion_matrices.csv", "predictions.csv", "selected_train_authors.csv"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def run_experiments(args: argparse.Namespace) -> None:
    data = np.load(args.embeddings_path, allow_pickle=False)
    x_train = data["x_train"]
    x_test = data["x_test"]
    train_author_ids = data["train_author_ids"]
    test_author_ids = data["test_author_ids"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(output_dir)

    summary_fields = [
        "task",
        "classifier",
        "train_fraction",
        "seed",
        "n_train",
        "metric",
        "value",
    ]
    per_class_fields = [
        "task",
        "classifier",
        "train_fraction",
        "seed",
        "n_train",
        "class_label",
        "precision",
        "recall",
        "f1",
        "support",
    ]
    confusion_fields = [
        "task",
        "classifier",
        "train_fraction",
        "seed",
        "n_train",
        "true_label",
        "predicted_label",
        "count",
    ]
    prediction_fields = [
        "task",
        "classifier",
        "train_fraction",
        "seed",
        "n_train",
        "author_id",
        "true_label",
        "predicted_label",
    ]
    selected_train_fields = [
        "task",
        "train_fraction",
        "seed",
        "author_id",
        "label",
    ]

    buffered_rows = defaultdict(list)

    for task_name, (train_key, test_key) in TASKS.items():
        y_train_labels = data[train_key]
        y_test_labels = data[test_key]
        label_encoder = LabelEncoder()
        label_encoder.fit(np.concatenate([y_train_labels, y_test_labels]))
        y_train = label_encoder.transform(y_train_labels)
        labels = label_encoder.classes_

        for train_fraction in args.train_fractions:
            for seed in args.seeds:
                train_indices = stratified_indices(y_train_labels, train_fraction, seed)
                x_train_subset = x_train[train_indices]
                y_train_subset = y_train[train_indices]
                buffered_rows["selected_train"].extend({
                        "task": task_name,
                        "train_fraction": train_fraction,
                        "seed": seed,
                        "author_id": train_author_ids[index],
                        "label": y_train_labels[index],
                    }
                    for index in train_indices
                )

                for classifier_name, classifier in make_classifiers(seed).items():
                    context = {
                        "task": task_name,
                        "classifier": classifier_name,
                        "train_fraction": train_fraction,
                        "seed": seed,
                        "n_train": len(train_indices),
                    }

                    classifier.fit(x_train_subset, y_train_subset)
                    y_pred_encoded = classifier.predict(x_test)
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)

                    for metric_name, value in evaluate(y_test_labels, y_pred).items():
                        buffered_rows["summary"].append({
                                **context,
                                "metric": metric_name,
                                "value": value,
                            })

                    report = classification_report(y_test_labels, y_pred, labels=labels, output_dict=True, zero_division=0)
                    buffered_rows["per_class"].extend(report_rows(report, context))
                    buffered_rows["confusion"].extend(confusion_rows(y_test_labels, y_pred, labels, context))
                    buffered_rows["predictions"].extend(prediction_rows(test_author_ids, y_test_labels, y_pred, context))
                    
                    print(
                        f"Done task={task_name} classifier={classifier_name} "
                        f"fraction={train_fraction} seed={seed}"
                    )

    append_rows(output_dir / "summary_metrics.csv", summary_fields, buffered_rows["summary"])
    append_rows(output_dir / "per_class_metrics.csv", per_class_fields, buffered_rows["per_class"])
    append_rows(output_dir / "confusion_matrices.csv", confusion_fields, buffered_rows["confusion"])
    append_rows(output_dir / "predictions.csv", prediction_fields, buffered_rows["predictions"])
    append_rows(output_dir / "selected_train_authors.csv", selected_train_fields, buffered_rows["selected_train"])


def main() -> None:
    args = parse_args()
    run_experiments(args)
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
