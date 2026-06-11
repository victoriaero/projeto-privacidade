from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.diff_priv import DEFAULT_DP_SEED, DELTA, SENSITIVITY, create_private_embeddings, private_embeddings_path
from src.evaluate_utility import DEFAULT_NEIGHBOR_KS, evaluate_embedding_utility
from src.generate_embeddings import DEFAULT_MODEL_NAME, EMBEDDINGS_DIR, PROCESSED_DATA_DIR, cache_name, generate_embeddings, short_model_name
from src.run_attacks import RESULTS_DIR, TRAIN_FRACTIONS, run_experiments
from src.summarize_results import summarize_results

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run the PAN15 privacy attack pipeline: generate cached author embeddings, optionally privatize them, then train and evaluate attacker classifiers."))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--aggregation", choices=["mean", "mean_std"], default="mean", help="How to aggregate document embeddings into one author vector.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional device: cuda or cpu.")
    parser.add_argument("--overwrite-embeddings", action="store_true", help="Regenerate embeddings even if a cache already exists.")
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DATA_DIR / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=PROCESSED_DATA_DIR / "test.csv")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=None, help="Defaults to outputs/<encoder>/<aggregation> for no-DP and outputs/<encoder>/<aggregation>/dp_eps... for DP.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 34, 55, 89])
    parser.add_argument("--train-fractions", type=float, nargs="+", default=TRAIN_FRACTIONS, help="Fractions of the official train split available to the attacker.")
    parser.add_argument("--privacy-mode", choices=["none", "dp"], default="none", help="Run attacks on original embeddings or on differentially private embeddings.")
    parser.add_argument("--dp-epsilons", type=float, nargs="+", default=[1.0], help="Epsilon values to run when --privacy-mode dp is selected.")
    parser.add_argument("--dp-delta", type=float, default=DELTA)
    parser.add_argument("--dp-sensitivity", type=float, default=SENSITIVITY, help="Sensitivity used by diffprivlib GaussianAnalytic.")
    parser.add_argument("--dp-seed", type=int, default=DEFAULT_DP_SEED, help="Seed used to generate DP noise.")
    parser.add_argument("--overwrite-private-embeddings", action="store_true", help="Regenerate DP embedding caches even if they already exist.")
    parser.add_argument("--utility-neighbor-ks", type=int, nargs="+", default=DEFAULT_NEIGHBOR_KS, help="Neighborhood sizes used by embedding utility metrics.")
    parser.add_argument("--skip-summary", action="store_true", help="Do not write aggregate privacy/utility comparison CSVs after DP runs.")
    parser.add_argument("--skip-attacks", action="store_true", help="Only generate/reuse embeddings; do not run classifiers.")
    return parser.parse_args()


def resolve_results_dir(results_dir: Path | None, model_name: str, aggregation: str) -> Path:
    if results_dir is not None:
        return results_dir
    return RESULTS_DIR / short_model_name(model_name) / aggregation


def resolve_dp_results_dir(base_results_dir: Path, epsilon: float) -> Path:
    epsilon_label = f"{epsilon:g}".replace(".", "p")
    return base_results_dir / f"dp_eps{epsilon_label}"


def write_run_config(output_dir: Path, args: argparse.Namespace, embeddings_path: Path, privacy_config: dict[str, object] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": args.model_name,
        "aggregation": args.aggregation,
        "batch_size": args.batch_size,
        "device": args.device,
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "embeddings_dir": str(args.embeddings_dir),
        "embeddings_path": str(embeddings_path),
        "results_dir": str(output_dir),
        "seeds": args.seeds,
        "train_fractions": args.train_fractions,
        "utility_neighbor_ks": args.utility_neighbor_ks,
        "skip_summary": args.skip_summary,
        "privacy": privacy_config or {"mode": "none"},
        "classifiers": [
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "mlp",
        ],
        "tasks": ["gender", "age"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_attacks_for_embeddings(args: argparse.Namespace, embeddings_path: Path, output_dir: Path, privacy_config: dict[str, object] | None = None) -> None:
    if args.skip_attacks:
        write_run_config(output_dir, args, embeddings_path, privacy_config)
        print(f"Skipping attacks. Run config saved to {output_dir}")
        return

    attack_args = argparse.Namespace(
        embeddings_path=embeddings_path,
        output_dir=output_dir,
        seeds=args.seeds,
        train_fractions=args.train_fractions,
    )
    run_experiments(attack_args)
    write_run_config(output_dir, args, embeddings_path, privacy_config)
    print(f"Pipeline finished. Results saved to {output_dir}")


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir, args.model_name, args.aggregation)

    embedding_args = argparse.Namespace(
        model_name=args.model_name,
        aggregation=args.aggregation,
        batch_size=args.batch_size,
        device=args.device,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.embeddings_dir,
        overwrite=args.overwrite_embeddings,
    )

    embeddings_path = generate_embeddings(embedding_args)

    expected_embeddings_path = args.embeddings_dir / cache_name(args.model_name, args.aggregation)
    if embeddings_path != expected_embeddings_path:
        raise RuntimeError(
            f"Unexpected embeddings path: {embeddings_path}. "
            f"Expected: {expected_embeddings_path}"
        )

    if args.privacy_mode == "none":
        run_attacks_for_embeddings(args, embeddings_path, results_dir, {"mode": "none"})
        return

    for epsilon in args.dp_epsilons:
        private_path = private_embeddings_path(
            embeddings_path,
            epsilon=epsilon,
            delta=args.dp_delta,
            sensitivity=args.dp_sensitivity,
            seed=args.dp_seed,
            output_dir=args.embeddings_dir,
        )
        create_private_embeddings(
            embeddings_path=embeddings_path,
            output_path=private_path,
            epsilon=epsilon,
            delta=args.dp_delta,
            sensitivity=args.dp_sensitivity,
            seed=args.dp_seed,
            overwrite=args.overwrite_private_embeddings,
        )
        dp_results_dir = resolve_dp_results_dir(results_dir, epsilon)
        evaluate_embedding_utility(
            original_embeddings=embeddings_path,
            private_embeddings=private_path,
            output_dir=dp_results_dir,
            neighbor_ks=args.utility_neighbor_ks,
        )
        privacy_config = {
            "mode": "dp",
            "source_embeddings_path": str(embeddings_path),
            "epsilon": float(epsilon),
            "delta": float(args.dp_delta),
            "sensitivity": float(args.dp_sensitivity),
            "seed": int(args.dp_seed),
            "private_embeddings_path": str(private_path),
        }
        run_attacks_for_embeddings(args, private_path, dp_results_dir, privacy_config)

    if not args.skip_summary:
        summarize_results(results_dir)


if __name__ == "__main__":
    main()
