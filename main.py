from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.generate_embeddings import DEFAULT_MODEL_NAME, EMBEDDINGS_DIR, PROCESSED_DATA_DIR, cache_name, generate_embeddings, short_model_name
from src.run_attacks import RESULTS_DIR, TRAIN_FRACTIONS, run_experiments

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=("Run the PAN15 privacy attack pipeline: generate cached author embeddings, then train and evaluate attacker classifiers."))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--aggregation", choices=["mean", "mean_std"], default="mean", help="How to aggregate document embeddings into one author vector.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional device: cuda or cpu.")
    parser.add_argument("--overwrite-embeddings", action="store_true", help="Regenerate embeddings even if a cache already exists.")
    parser.add_argument("--train-csv", type=Path, default=PROCESSED_DATA_DIR / "train.csv")
    parser.add_argument("--test-csv", type=Path, default=PROCESSED_DATA_DIR / "test.csv")
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=None, help="Defaults to outputs/<encoder>/<aggregation>.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 34, 55, 89])
    parser.add_argument("--train-fractions", type=float, nargs="+", default=TRAIN_FRACTIONS, help="Fractions of the official train split available to the attacker.")
    parser.add_argument("--skip-attacks", action="store_true", help="Only generate/reuse embeddings; do not run classifiers.")
    return parser.parse_args()


def resolve_results_dir(results_dir: Path | None, model_name: str, aggregation: str) -> Path:
    if results_dir is not None:
        return results_dir
    return RESULTS_DIR / short_model_name(model_name) / aggregation


def write_run_config(output_dir: Path, args: argparse.Namespace, embeddings_path: Path) -> None:
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
        "classifiers": [
            "logistic_regression",
            "random_forest",
            "hist_gradient_boosting",
            "mlp",
        ],
        "tasks": ["gender", "age"],
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_dir = resolve_results_dir(args.results_dir, args.model_name, args.aggregation)

    embedding_args = argparse.Namespace(model_name=args.model_name, aggregation=args.aggregation, batch_size=args.batch_size, device=args.device, train_csv=args.train_csv, test_csv=args.test_csv, output_dir=args.embeddings_dir, overwrite=args.overwrite_embeddings)

    embeddings_path = generate_embeddings(embedding_args)

    if args.skip_attacks:
        write_run_config(results_dir, args, embeddings_path)
        print("Skipping attacks.")
        return

    expected_embeddings_path = args.embeddings_dir / cache_name(args.model_name, args.aggregation)
    if embeddings_path != expected_embeddings_path:
        raise RuntimeError(
            f"Unexpected embeddings path: {embeddings_path}. "
            f"Expected: {expected_embeddings_path}"
        )

    attack_args = argparse.Namespace(embeddings_path=embeddings_path, output_dir=results_dir, seeds=args.seeds, train_fractions=args.train_fractions)
    run_experiments(attack_args)
    write_run_config(results_dir, args, embeddings_path)
    print(f"Pipeline finished. Results saved to {results_dir}")


if __name__ == "__main__":
    main()
