from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ATTACK_METRICS = ["accuracy", "balanced_accuracy", "f1_macro"]
CHANCE_BALANCED_ACCURACY = {
    "gender": 0.5,
    "age": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize privacy and utility results across no-DP and DP runs.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Base directory containing no-DP CSVs and dp_eps* subdirectories.")
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_privacy_config(run_dir: Path) -> dict[str, object]:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {"mode": "none"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config.get("privacy", {"mode": "none"})


def dp_run_dirs(results_dir: Path) -> list[Path]:
    return sorted(path for path in results_dir.glob("dp_eps*") if (path / "summary_metrics.csv").exists())


def aggregate_attack_rows(rows: list[dict[str, str]], privacy: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    for row in rows:
        metric = row["metric"]
        if metric not in ATTACK_METRICS:
            continue
        key = (row["task"], row["classifier"], row["train_fraction"], metric)
        grouped.setdefault(key, []).append(float(row["value"]))

    output = []
    for (task, classifier, train_fraction, metric), values in sorted(grouped.items()):
        output.append({
            "privacy_mode": privacy.get("mode", "none"),
            "epsilon": privacy.get("epsilon", ""),
            "delta": privacy.get("delta", ""),
            "sensitivity": privacy.get("sensitivity", ""),
            "dp_seed": privacy.get("seed", ""),
            "task": task,
            "classifier": classifier,
            "train_fraction": train_fraction,
            "metric": metric,
            "mean": mean(values),
            "std": std(values),
            "n": len(values),
        })
    return output


def aggregate_attacks(results_dir: Path) -> list[dict[str, object]]:
    rows = []
    baseline_summary = results_dir / "summary_metrics.csv"
    if baseline_summary.exists():
        rows.extend(aggregate_attack_rows(read_csv_rows(baseline_summary), {"mode": "none"}))

    for run_dir in dp_run_dirs(results_dir):
        rows.extend(aggregate_attack_rows(read_csv_rows(run_dir / "summary_metrics.csv"), load_privacy_config(run_dir)))
    return rows


def baseline_lookup(attack_rows: list[dict[str, object]]) -> dict[tuple[str, str, str, str], float]:
    lookup = {}
    for row in attack_rows:
        if row["privacy_mode"] != "none":
            continue
        key = (str(row["task"]), str(row["classifier"]), str(row["train_fraction"]), str(row["metric"]))
        lookup[key] = float(row["mean"])
    return lookup


def add_privacy_comparison(attack_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = baseline_lookup(attack_rows)
    output = []
    for row in attack_rows:
        enriched = dict(row)
        key = (str(row["task"]), str(row["classifier"]), str(row["train_fraction"]), str(row["metric"]))
        baseline_value = lookup.get(key, float("nan"))
        current_value = float(row["mean"])
        enriched["baseline_mean"] = baseline_value
        enriched["privacy_gain_vs_baseline"] = baseline_value - current_value if np.isfinite(baseline_value) else float("nan")
        enriched["relative_attack_reduction"] = (baseline_value - current_value) / baseline_value if baseline_value else float("nan")

        chance = CHANCE_BALANCED_ACCURACY.get(str(row["task"]), float("nan")) if row["metric"] == "balanced_accuracy" else float("nan")
        enriched["chance_level"] = chance
        enriched["attack_advantage_over_chance"] = current_value - chance if np.isfinite(chance) else float("nan")
        output.append(enriched)
    return output


def aggregate_privacy_by_epsilon(attack_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in attack_rows:
        if row["privacy_mode"] != "dp":
            continue
        key = (str(row["epsilon"]), str(row["task"]), str(row["metric"]))
        grouped.setdefault(key, []).append(row)

    output = []
    for (epsilon, task, metric), rows in sorted(grouped.items(), key=lambda item: (float(item[0][0]), item[0][1], item[0][2])):
        output.append({
            "epsilon": epsilon,
            "task": task,
            "metric": metric,
            "mean_attack_score": mean([float(row["mean"]) for row in rows]),
            "mean_baseline_score": mean([float(row["baseline_mean"]) for row in rows]),
            "mean_privacy_gain_vs_baseline": mean([float(row["privacy_gain_vs_baseline"]) for row in rows]),
            "mean_relative_attack_reduction": mean([float(row["relative_attack_reduction"]) for row in rows]),
            "mean_attack_advantage_over_chance": mean([float(row["attack_advantage_over_chance"]) for row in rows if np.isfinite(float(row["attack_advantage_over_chance"]))]),
            "n_groups": len(rows),
        })
    return output


def load_utility_rows(run_dir: Path) -> list[dict[str, object]]:
    utility_path = run_dir / "utility_metrics.csv"
    if not utility_path.exists():
        return []
    privacy = load_privacy_config(run_dir)
    rows = []
    for row in read_csv_rows(utility_path):
        rows.append({
            "epsilon": privacy.get("epsilon", ""),
            "delta": privacy.get("delta", ""),
            "sensitivity": privacy.get("sensitivity", ""),
            "dp_seed": privacy.get("seed", ""),
            "split": row["split"],
            "metric": row["metric"],
            "k": row["k"],
            "value": float(row["value"]),
        })
    return rows


def aggregate_utility(results_dir: Path) -> list[dict[str, object]]:
    rows = []
    for run_dir in dp_run_dirs(results_dir):
        rows.extend(load_utility_rows(run_dir))
    return rows


def compact_tradeoff_summary(privacy_rows: list[dict[str, object]], utility_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    privacy_lookup = {}
    for row in privacy_rows:
        if row["metric"] == "balanced_accuracy":
            privacy_lookup[(str(row["epsilon"]), str(row["task"]))] = row

    utility_lookup = {}
    for row in utility_rows:
        if row["split"] == "test" and row["metric"] in {"cosine_mean", "l2_distance_mean", "relative_l2_distance_mean"}:
            utility_lookup[(str(row["epsilon"]), row["metric"])] = float(row["value"])
        if row["split"] == "test" and row["metric"] == "neighbor_overlap" and str(row["k"]) == "10":
            utility_lookup[(str(row["epsilon"]), "neighbor_overlap@10")] = float(row["value"])

    epsilons = sorted({key[0] for key in privacy_lookup} | {key[0] for key in utility_lookup}, key=float)
    output = []
    for epsilon in epsilons:
        gender = privacy_lookup.get((epsilon, "gender"), {})
        age = privacy_lookup.get((epsilon, "age"), {})
        output.append({
            "epsilon": epsilon,
            "gender_balanced_accuracy": gender.get("mean_attack_score", ""),
            "gender_privacy_gain": gender.get("mean_privacy_gain_vs_baseline", ""),
            "gender_advantage_over_chance": gender.get("mean_attack_advantage_over_chance", ""),
            "age_balanced_accuracy": age.get("mean_attack_score", ""),
            "age_privacy_gain": age.get("mean_privacy_gain_vs_baseline", ""),
            "age_advantage_over_chance": age.get("mean_attack_advantage_over_chance", ""),
            "test_cosine_mean": utility_lookup.get((epsilon, "cosine_mean"), ""),
            "test_l2_distance_mean": utility_lookup.get((epsilon, "l2_distance_mean"), ""),
            "test_relative_l2_distance_mean": utility_lookup.get((epsilon, "relative_l2_distance_mean"), ""),
            "test_neighbor_overlap@10": utility_lookup.get((epsilon, "neighbor_overlap@10"), ""),
        })
    return output


def summarize_results(results_dir: Path) -> None:
    attack_rows = add_privacy_comparison(aggregate_attacks(results_dir))
    privacy_rows = aggregate_privacy_by_epsilon(attack_rows)
    utility_rows = aggregate_utility(results_dir)
    tradeoff_rows = compact_tradeoff_summary(privacy_rows, utility_rows)

    write_csv_rows(
        results_dir / "attack_comparison_metrics.csv",
        [
            "privacy_mode", "epsilon", "delta", "sensitivity", "dp_seed", "task", "classifier", "train_fraction",
            "metric", "mean", "std", "n", "baseline_mean", "privacy_gain_vs_baseline", "relative_attack_reduction",
            "chance_level", "attack_advantage_over_chance",
        ],
        attack_rows,
    )
    write_csv_rows(
        results_dir / "privacy_by_epsilon.csv",
        [
            "epsilon", "task", "metric", "mean_attack_score", "mean_baseline_score", "mean_privacy_gain_vs_baseline",
            "mean_relative_attack_reduction", "mean_attack_advantage_over_chance", "n_groups",
        ],
        privacy_rows,
    )
    write_csv_rows(
        results_dir / "utility_by_epsilon.csv",
        ["epsilon", "delta", "sensitivity", "dp_seed", "split", "metric", "k", "value"],
        utility_rows,
    )
    write_csv_rows(
        results_dir / "privacy_utility_tradeoff.csv",
        [
            "epsilon", "gender_balanced_accuracy", "gender_privacy_gain", "gender_advantage_over_chance",
            "age_balanced_accuracy", "age_privacy_gain", "age_advantage_over_chance", "test_cosine_mean",
            "test_l2_distance_mean", "test_relative_l2_distance_mean", "test_neighbor_overlap@10",
        ],
        tradeoff_rows,
    )
    print(f"Saved comparison summaries to {results_dir}")


def main() -> None:
    args = parse_args()
    summarize_results(args.results_dir)


if __name__ == "__main__":
    main()
