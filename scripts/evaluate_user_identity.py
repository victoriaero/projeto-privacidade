from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENT_SEPARATOR = "\n\n"
LEGACY_DOCUMENT_SEPARATOR = " "

DEFAULT_EPSILONS = ["none", "0.5", "1", "5", "10", "20", "50", "100"]
DEFAULT_SEEDS = [13, 21, 34, 55, 89]


def safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def document_cache_name(dataset_name: str, model_name: str) -> str:
    safe_dataset = safe_name(dataset_name)
    safe_model = model_name.replace("/", "__")
    return f"{safe_dataset}_{safe_model}_documents.npz"


def split_documents(text: str) -> list[str]:
    text = str(text)

    if DOCUMENT_SEPARATOR in text:
        return [
            document.strip()
            for document in text.split(DOCUMENT_SEPARATOR)
            if document.strip()
        ]

    return [
        document.strip()
        for document in text.split(LEGACY_DOCUMENT_SEPARATOR)
        if document.strip()
    ]


def read_processed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    expected_columns = {"author_id", "gender", "age_group", "text"}
    missing_columns = expected_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    return df


def build_document_table(
    train_csv: Path,
    test_csv: Path,
    max_docs_per_author: int | None,
    seed: int,
) -> pd.DataFrame:
    frames = []

    for split, path in [("train", train_csv), ("test", test_csv)]:
        df = read_processed_csv(path)

        rows = []

        rng = np.random.default_rng(seed)

        for row in df.itertuples(index=False):
            documents = split_documents(row.text)

            if max_docs_per_author is not None and len(documents) > max_docs_per_author:
                selected = rng.choice(
                    np.arange(len(documents)),
                    size=max_docs_per_author,
                    replace=False,
                )
                selected = sorted(selected.tolist())
                documents = [documents[i] for i in selected]

            for doc_id, document in enumerate(documents):
                rows.append(
                    {
                        "split": split,
                        "author_id": str(row.author_id),
                        "gender": str(row.gender),
                        "age_group": str(row.age_group),
                        "doc_id": doc_id,
                        "text": document,
                    }
                )

        frames.append(pd.DataFrame(rows))

    documents_df = pd.concat(frames, ignore_index=True)

    if documents_df.empty:
        raise ValueError("No documents were found after splitting the processed CSVs.")

    return documents_df


def load_or_generate_document_embeddings(
    documents_df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    device: str | None,
    output_dir: Path,
    overwrite: bool,
) -> tuple[pd.DataFrame, np.ndarray, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / document_cache_name(dataset_name, model_name)

    if cache_path.exists() and not overwrite:
        print(f"Loading cached document embeddings: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        cached_df = pd.DataFrame(
            {
                "split": data["split"].astype(str),
                "author_id": data["author_id"].astype(str),
                "gender": data["gender"].astype(str),
                "age_group": data["age_group"].astype(str),
                "doc_id": data["doc_id"],
                "text": data["text"].astype(str),
            }
        )

        return cached_df, data["embeddings"].astype(np.float32), cache_path

    print("Generating document-level embeddings.")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    texts = documents_df["text"].astype(str).tolist()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype(np.float32)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        split=documents_df["split"].to_numpy(dtype=str),
        author_id=documents_df["author_id"].to_numpy(dtype=str),
        gender=documents_df["gender"].to_numpy(dtype=str),
        age_group=documents_df["age_group"].to_numpy(dtype=str),
        doc_id=documents_df["doc_id"].to_numpy(),
        text=documents_df["text"].to_numpy(dtype=str),
    )

    metadata = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "cache_path": str(cache_path),
        "n_documents": int(len(documents_df)),
        "embedding_shape": list(embeddings.shape),
    }

    cache_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved document embeddings to {cache_path}")

    return documents_df, embeddings, cache_path


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def aggregate_indices(embeddings: np.ndarray, indices: Iterable[int]) -> np.ndarray:
    idx = np.array(list(indices), dtype=int)

    if len(idx) == 0:
        raise ValueError("Cannot aggregate an empty set of indices.")

    return embeddings[idx].mean(axis=0)


def parse_epsilon(value: str) -> float | None:
    if str(value).lower() in {"none", "no", "original"}:
        return None

    return float(value)


def add_dp_noise(
    x: np.ndarray,
    epsilon: float | None,
    delta: float,
    sensitivity: float,
    seed: int,
    use_diffprivlib: bool,
) -> np.ndarray:
    if epsilon is None:
        return x.copy()

    if use_diffprivlib:
        try:
            from diffprivlib.mechanisms import GaussianAnalytic

            try:
                mech = GaussianAnalytic(
                    epsilon=epsilon,
                    delta=delta,
                    sensitivity=sensitivity,
                    random_state=seed,
                )
            except TypeError:
                mech = GaussianAnalytic(
                    epsilon=epsilon,
                    delta=delta,
                    sensitivity=sensitivity,
                )

            flat = x.astype(float).ravel()
            private = np.empty_like(flat)

            for i, value in enumerate(flat):
                private[i] = mech.randomise(float(value))

            return private.reshape(x.shape).astype(np.float32)

        except ImportError:
            print("diffprivlib not found. Falling back to NumPy Gaussian approximation.")

    rng = np.random.default_rng(seed)
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape)

    return (x + noise).astype(np.float32)


def split_author_documents(
    documents_df: pd.DataFrame,
    min_docs_per_author: int,
    split_seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(split_seed)

    author_splits = {}

    for author_id, group in documents_df.groupby("author_id"):
        doc_indices = group.index.to_numpy()

        if len(doc_indices) < min_docs_per_author:
            continue

        shuffled = doc_indices.copy()
        rng.shuffle(shuffled)

        cut = len(shuffled) // 2

        if cut == 0 or cut == len(shuffled):
            continue

        author_splits[str(author_id)] = {
            "gallery": shuffled[:cut],
            "query": shuffled[cut:],
        }

    return author_splits


def build_user_halves(
    documents_df: pd.DataFrame,
    embeddings: np.ndarray,
    min_docs_per_author: int,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, np.ndarray]]]:
    author_splits = split_author_documents(
        documents_df=documents_df,
        min_docs_per_author=min_docs_per_author,
        split_seed=split_seed,
    )

    if not author_splits:
        raise ValueError(
            "No authors remained after filtering. "
            "Try lowering --min-docs-per-author."
        )

    author_ids = sorted(author_splits.keys())

    gallery_embeddings = []
    query_embeddings = []

    for author_id in author_ids:
        gallery_embeddings.append(
            aggregate_indices(embeddings, author_splits[author_id]["gallery"])
        )
        query_embeddings.append(
            aggregate_indices(embeddings, author_splits[author_id]["query"])
        )

    gallery_author_ids = np.array(author_ids, dtype=str)
    query_author_ids = np.array(author_ids, dtype=str)

    return (
        gallery_author_ids,
        np.vstack(gallery_embeddings).astype(np.float32),
        query_author_ids,
        np.vstack(query_embeddings).astype(np.float32),
        author_splits,
    )


def ranking_metrics(
    query_embeddings: np.ndarray,
    query_author_ids: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_author_ids: np.ndarray,
    top_ks: list[int],
    chunk_size: int,
) -> dict[str, float]:
    query_embeddings = l2_normalize(query_embeddings.astype(float))
    gallery_embeddings = l2_normalize(gallery_embeddings.astype(float))

    gallery_position = {
        str(author_id): i
        for i, author_id in enumerate(gallery_author_ids)
    }

    ranks = []
    positive_cosines = []
    best_negative_cosines = []

    n_queries = len(query_author_ids)

    for start in range(0, n_queries, chunk_size):
        end = min(start + chunk_size, n_queries)

        q = query_embeddings[start:end]
        q_author_ids = query_author_ids[start:end]

        sim = q @ gallery_embeddings.T

        for local_i, author_id in enumerate(q_author_ids):
            correct_idx = gallery_position[str(author_id)]
            correct_score = sim[local_i, correct_idx]

            rank = int(np.sum(sim[local_i] > correct_score)) + 1
            ranks.append(rank)
            positive_cosines.append(float(correct_score))

            row = sim[local_i].copy()
            row[correct_idx] = -np.inf
            best_negative_cosines.append(float(np.max(row)))

    ranks = np.array(ranks, dtype=float)

    output = {
        "n_queries": int(n_queries),
        "n_gallery": int(len(gallery_author_ids)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "positive_cosine_mean": float(np.mean(positive_cosines)),
        "best_negative_cosine_mean": float(np.mean(best_negative_cosines)),
        "cosine_margin_mean": float(
            np.mean(np.array(positive_cosines) - np.array(best_negative_cosines))
        ),
    }

    for k in top_ks:
        output[f"top{k}_acc"] = float(np.mean(ranks <= k))

    return output


def make_message_queries(
    author_splits: dict[str, dict[str, np.ndarray]],
    embeddings: np.ndarray,
    max_query_docs_per_author: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    query_embeddings = []
    query_author_ids = []

    for author_id, parts in author_splits.items():
        query_indices = parts["query"]

        if max_query_docs_per_author > 0 and len(query_indices) > max_query_docs_per_author:
            query_indices = rng.choice(
                query_indices,
                size=max_query_docs_per_author,
                replace=False,
            )

        for idx in query_indices:
            query_embeddings.append(embeddings[idx])
            query_author_ids.append(author_id)

    return (
        np.array(query_author_ids, dtype=str),
        np.vstack(query_embeddings).astype(np.float32),
    )


def apply_dp_mode(
    gallery_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    epsilon: float | None,
    delta: float,
    sensitivity: float,
    seed: int,
    dp_mode: str,
    use_diffprivlib: bool,
) -> tuple[np.ndarray, np.ndarray]:
    gallery_out = gallery_embeddings.copy()
    query_out = query_embeddings.copy()

    if epsilon is None:
        return gallery_out, query_out

    if dp_mode in {"gallery_only", "both"}:
        gallery_out = add_dp_noise(
            gallery_out,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            seed=seed + 100_000,
            use_diffprivlib=use_diffprivlib,
        )

    if dp_mode in {"query_only", "both"}:
        query_out = add_dp_noise(
            query_out,
            epsilon=epsilon,
            delta=delta,
            sensitivity=sensitivity,
            seed=seed + 200_000,
            use_diffprivlib=use_diffprivlib,
        )

    return gallery_out, query_out


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


def run_analysis(args: argparse.Namespace) -> None:
    base_results_dir = (
        args.output_dir
        / safe_name(args.dataset_name)
        / short_model_name(args.model_name)
        / args.aggregation
    )

    identity_dir = base_results_dir / "user_identity"
    identity_dir.mkdir(parents=True, exist_ok=True)

    output_path = identity_dir / "user_identity_metrics.csv"

    if output_path.exists() and args.overwrite_results:
        output_path.unlink()

    documents_df = build_document_table(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        max_docs_per_author=args.max_docs_per_author,
        seed=args.document_seed,
    )

    documents_df, embeddings, cache_path = load_or_generate_document_embeddings(
        documents_df=documents_df,
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.embeddings_dir,
        overwrite=args.overwrite_document_embeddings,
    )

    print(f"Documents: {len(documents_df)}")
    print(f"Authors: {documents_df['author_id'].nunique()}")
    print(f"Embedding cache: {cache_path}")

    all_rows = []

    for split_seed in args.seeds:
        print(f"\nRunning split seed={split_seed}")

        (
            gallery_author_ids,
            gallery_embeddings,
            query_author_ids,
            query_embeddings,
            author_splits,
        ) = build_user_halves(
            documents_df=documents_df,
            embeddings=embeddings,
            min_docs_per_author=args.min_docs_per_author,
            split_seed=split_seed,
        )

        print(f"Authors after min-doc filter: {len(gallery_author_ids)}")

        for epsilon_label in args.epsilons:
            epsilon = parse_epsilon(epsilon_label)

            for dp_mode in args.dp_modes:
                if epsilon is None and dp_mode != "none":
                    continue

                effective_dp_mode = "none" if epsilon is None else dp_mode

                gallery_eval, query_eval = apply_dp_mode(
                    gallery_embeddings=gallery_embeddings,
                    query_embeddings=query_embeddings,
                    epsilon=epsilon,
                    delta=args.dp_delta,
                    sensitivity=args.dp_sensitivity,
                    seed=args.dp_seed + split_seed,
                    dp_mode=effective_dp_mode,
                    use_diffprivlib=args.use_diffprivlib,
                )

                metrics = ranking_metrics(
                    query_embeddings=query_eval,
                    query_author_ids=query_author_ids,
                    gallery_embeddings=gallery_eval,
                    gallery_author_ids=gallery_author_ids,
                    top_ks=args.top_ks,
                    chunk_size=args.chunk_size,
                )

                row = {
                    "dataset": args.dataset_name,
                    "model_name": args.model_name,
                    "aggregation": args.aggregation,
                    "analysis": "user_to_user_halves",
                    "epsilon": "none" if epsilon is None else float(epsilon),
                    "dp_mode": effective_dp_mode,
                    "split_seed": split_seed,
                    "min_docs_per_author": args.min_docs_per_author,
                    "max_docs_per_author": args.max_docs_per_author
                    if args.max_docs_per_author is not None
                    else "",
                    "delta": args.dp_delta if epsilon is not None else "",
                    "sensitivity": args.dp_sensitivity if epsilon is not None else "",
                    **metrics,
                }

                all_rows.append(row)

                print(
                    "user_to_user",
                    f"eps={row['epsilon']}",
                    f"mode={effective_dp_mode}",
                    f"top1={metrics.get('top1_acc', np.nan):.4f}",
                    f"top5={metrics.get('top5_acc', np.nan):.4f}",
                    f"mrr={metrics['mrr']:.4f}",
                )

                if args.run_message_to_user:
                    message_author_ids, message_embeddings = make_message_queries(
                        author_splits=author_splits,
                        embeddings=embeddings,
                        max_query_docs_per_author=args.max_query_docs_per_author,
                        seed=args.dp_seed + split_seed,
                    )

                    gallery_msg_eval, message_eval = apply_dp_mode(
                        gallery_embeddings=gallery_embeddings,
                        query_embeddings=message_embeddings,
                        epsilon=epsilon,
                        delta=args.dp_delta,
                        sensitivity=args.dp_sensitivity,
                        seed=args.dp_seed + split_seed + 777,
                        dp_mode=effective_dp_mode,
                        use_diffprivlib=args.use_diffprivlib,
                    )

                    msg_metrics = ranking_metrics(
                        query_embeddings=message_eval,
                        query_author_ids=message_author_ids,
                        gallery_embeddings=gallery_msg_eval,
                        gallery_author_ids=gallery_author_ids,
                        top_ks=args.top_ks,
                        chunk_size=args.chunk_size,
                    )

                    msg_row = {
                        "dataset": args.dataset_name,
                        "model_name": args.model_name,
                        "aggregation": args.aggregation,
                        "analysis": "message_to_user",
                        "epsilon": "none" if epsilon is None else float(epsilon),
                        "dp_mode": effective_dp_mode,
                        "split_seed": split_seed,
                        "min_docs_per_author": args.min_docs_per_author,
                        "max_docs_per_author": args.max_docs_per_author
                        if args.max_docs_per_author is not None
                        else "",
                        "delta": args.dp_delta if epsilon is not None else "",
                        "sensitivity": args.dp_sensitivity if epsilon is not None else "",
                        **msg_metrics,
                    }

                    all_rows.append(msg_row)

                    print(
                        "message_to_user",
                        f"eps={msg_row['epsilon']}",
                        f"mode={effective_dp_mode}",
                        f"top1={msg_metrics.get('top1_acc', np.nan):.4f}",
                        f"top5={msg_metrics.get('top5_acc', np.nan):.4f}",
                        f"mrr={msg_metrics['mrr']:.4f}",
                    )

        append_csv(output_path, all_rows)
        all_rows = []

    config = vars(args).copy()
    config["output_path"] = str(output_path)
    config["document_embeddings_cache"] = str(cache_path)

    (identity_dir / "user_identity_config.json").write_text(
        json.dumps(config, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\nSaved user identity metrics to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether user embeddings still encode user identity "
            "under differentially private perturbations."
        )
    )

    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--aggregation", default="mean")

    parser.add_argument(
        "--train-csv",
        type=Path,
        default=PROCESSED_DATA_DIR / "train.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=PROCESSED_DATA_DIR / "test.csv",
    )

    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--document-seed", type=int, default=2026)

    parser.add_argument(
        "--epsilons",
        nargs="+",
        default=DEFAULT_EPSILONS,
        help='Use "none" plus numeric epsilons, e.g. none 0.5 1 5 10 20 50 100.',
    )

    parser.add_argument(
        "--dp-modes",
        nargs="+",
        default=["query_only", "both"],
        choices=["none", "query_only", "gallery_only", "both"],
        help=(
            "query_only simulates protected embeddings being matched against "
            "a clean auxiliary gallery. both simulates both sides protected."
        ),
    )

    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-sensitivity", type=float, default=1.0)
    parser.add_argument("--dp-seed", type=int, default=2026)

    parser.add_argument(
        "--use-diffprivlib",
        action="store_true",
        help=(
            "Use diffprivlib GaussianAnalytic, matching the current project more closely. "
            "Without this flag, uses a faster NumPy Gaussian approximation."
        ),
    )

    parser.add_argument(
        "--min-docs-per-author",
        type=int,
        default=4,
        help="Minimum number of documents required so the author can be split into two halves.",
    )

    parser.add_argument(
        "--max-docs-per-author",
        type=int,
        default=None,
        help="Optional cap on documents per author before embedding, useful for Blog Authorship.",
    )

    parser.add_argument(
        "--run-message-to-user",
        action="store_true",
        help="Also evaluate whether individual messages retrieve their correct user profile.",
    )

    parser.add_argument(
        "--max-query-docs-per-author",
        type=int,
        default=5,
        help="For message-to-user retrieval, max query documents sampled per author.",
    )

    parser.add_argument("--top-ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--chunk-size", type=int, default=512)

    parser.add_argument("--overwrite-document-embeddings", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()