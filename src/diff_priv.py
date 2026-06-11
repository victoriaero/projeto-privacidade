from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic

EPSILON = 1.0
DELTA = 1e-5
SENSITIVITY = 1.0
DEFAULT_DP_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate differentially private embedding caches.")
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[1.0, 5.0, 10.0])
    parser.add_argument("--delta", type=float, default=DELTA)
    parser.add_argument("--sensitivity", type=float, default=SENSITIVITY)
    parser.add_argument("--seed", type=int, default=DEFAULT_DP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def format_float_for_filename(value: float) -> str:
    return f"{value:g}".replace(".", "p")

def private_embeddings_path(embeddings_path: Path, epsilon: float, delta: float, sensitivity: float, seed: int, output_dir: Path | None = None) -> Path:
    target_dir = output_dir if output_dir is not None else embeddings_path.parent
    suffix = (f"dp_eps{format_float_for_filename(epsilon)}"
        f"_delta{format_float_for_filename(delta)}"
        f"_sens{format_float_for_filename(sensitivity)}"
        f"_seed{seed}")
    return target_dir / f"{embeddings_path.stem}_{suffix}.npz"

def create_mechanism(epsilon: float, delta: float, sensitivity: float, seed: int):
    try:
        return GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity, random_state=seed)
    except TypeError:
        return GaussianAnalytic(epsilon=epsilon, delta=delta, sensitivity=sensitivity)

def create_private_embeddings(embeddings_path: Path, output_path: Path, epsilon: float = EPSILON, delta: float = DELTA, sensitivity: float = SENSITIVITY, seed: int = DEFAULT_DP_SEED, overwrite: bool = False) -> Path:
    if output_path.exists() and not overwrite:
        print(f"Private embedding cache already exists: {output_path}")
        print("Use --overwrite to regenerate it.")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(embeddings_path, allow_pickle=True)

    x_train = data["x_train"]
    x_test = data["x_test"]

    mech = create_mechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity, seed=seed)

    x_train_private = np.empty_like(x_train)

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            x_train_private[i, j] = mech.randomise(float(x_train[i, j]))

    x_test_private = np.empty_like(x_test)

    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            x_test_private[i, j] = mech.randomise(float(x_test[i, j]))

    np.savez_compressed(
        output_path,
        x_train=x_train_private,
        x_test=x_test_private,
        train_author_ids=data["train_author_ids"],
        test_author_ids=data["test_author_ids"],
        y_train_gender=data["y_train_gender"],
        y_test_gender=data["y_test_gender"],
        y_train_age=data["y_train_age"],
        y_test_age=data["y_test_age"],
    )

    metadata = {
        "source_embeddings_path": str(embeddings_path),
        "output_path": str(output_path),
        "privacy_enabled": True,
        "mechanism": "diffprivlib.mechanisms.GaussianAnalytic",
        "epsilon": float(epsilon),
        "delta": float(delta),
        "sensitivity": float(sensitivity),
        "seed": int(seed),
        "x_train_shape": list(x_train_private.shape),
        "x_test_shape": list(x_test_private.shape),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved private embeddings to {output_path}")
    print(f"Saved private metadata to {output_path.with_suffix('.json')}")
    return output_path

def main() -> None:
    args = parse_args()
    for eps in args.epsilons:
        output_path = private_embeddings_path(args.embeddings_path, epsilon=eps, delta=args.delta, sensitivity=args.sensitivity, seed=args.seed, output_dir=args.output_dir)
        create_private_embeddings(embeddings_path=args.embeddings_path, output_path=output_path, epsilon=eps, delta=args.delta, sensitivity=args.sensitivity, seed=args.seed, overwrite=args.overwrite)

if __name__ == "__main__":
    main()
