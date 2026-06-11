from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_NEIGHBOR_KS = [5, 10, 20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate intrinsic utility of private embeddings against original embeddings.")
    parser.add_argument("--original-embeddings", type=Path, required=True)
    parser.add_argument("--private-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neighbor-ks", type=int, nargs="+", default=DEFAULT_NEIGHBOR_KS)
    return parser.parse_args()


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(denominator, 1e-12)


def summarize_values(values: np.ndarray, prefix: str) -> list[dict[str, object]]:
    return [
        {"metric": f"{prefix}_mean", "value": float(np.mean(values))},
        {"metric": f"{prefix}_std", "value": float(np.std(values))},
        {"metric": f"{prefix}_min", "value": float(np.min(values))},
        {"metric": f"{prefix}_max", "value": float(np.max(values))},
    ]


def cosine_per_row(original: np.ndarray, private: np.ndarray) -> np.ndarray:
    numerator = np.sum(original * private, axis=1)
    denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(private, axis=1)
    return safe_divide(numerator, denominator)


def neighbor_indices(values: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = safe_divide(values, norms)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(k, values.shape[0] - 1)
    if k <= 0:
        return np.empty((values.shape[0], 0), dtype=int)
    partition = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(values.shape[0])[:, None]
    order = np.argsort(-similarity[row_indices, partition], axis=1)
    return partition[row_indices, order]


def neighbor_overlap_at_k(original: np.ndarray, private: np.ndarray, k: int) -> float:
    original_neighbors = neighbor_indices(original, k)
    private_neighbors = neighbor_indices(private, k)
    if original_neighbors.shape[1] == 0:
        return 0.0

    overlaps = []
    for original_row, private_row in zip(original_neighbors, private_neighbors):
        overlaps.append(len(set(original_row).intersection(private_row)) / original_neighbors.shape[1])
    return float(np.mean(overlaps))


def split_utility_rows(original: np.ndarray, private: np.ndarray, split: str, neighbor_ks: list[int]) -> list[dict[str, object]]:
    if original.shape != private.shape:
        raise ValueError(f"Shape mismatch for {split}: original={original.shape}, private={private.shape}")

    delta = private - original
    original_norm = np.linalg.norm(original, axis=1)
    private_norm = np.linalg.norm(private, axis=1)
    delta_norm = np.linalg.norm(delta, axis=1)

    metric_rows = []
    metric_rows.extend(summarize_values(cosine_per_row(original, private), "cosine"))
    metric_rows.extend(summarize_values(delta_norm, "l2_distance"))
    metric_rows.extend(summarize_values(np.abs(delta), "absolute_delta"))
    metric_rows.extend(summarize_values(original_norm, "original_norm"))
    metric_rows.extend(summarize_values(private_norm, "private_norm"))
    metric_rows.extend(summarize_values(safe_divide(delta_norm, original_norm), "relative_l2_distance"))
    metric_rows.extend(summarize_values(safe_divide(private_norm, original_norm), "private_to_original_norm_ratio"))

    rows = []
    for row in metric_rows:
        rows.append({"split": split, "metric": row["metric"], "k": "", "value": row["value"]})

    for k in neighbor_ks:
        rows.append({
            "split": split,
            "metric": "neighbor_overlap",
            "k": int(k),
            "value": neighbor_overlap_at_k(original, private, k),
        })

    return rows


def evaluate_embedding_utility(original_embeddings: Path, private_embeddings: Path, output_dir: Path, neighbor_ks: list[int] | None = None) -> Path:
    neighbor_ks = neighbor_ks or DEFAULT_NEIGHBOR_KS
    original_data = np.load(original_embeddings, allow_pickle=False)
    private_data = np.load(private_embeddings, allow_pickle=False)

    rows = []
    for split in ["train", "test"]:
        rows.extend(split_utility_rows(original_data[f"x_{split}"].astype(float), private_data[f"x_{split}"].astype(float), split, neighbor_ks))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "utility_metrics.csv"
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "metric", "k", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved utility metrics to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    evaluate_embedding_utility(args.original_embeddings, args.private_embeddings, args.output_dir, args.neighbor_ks)


if __name__ == "__main__":
    main()
