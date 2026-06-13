from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
DOCUMENT_SEPARATOR = "\n<DOC_SEP>\n"
LEGACY_DOCUMENT_SEPARATOR = "  "
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate cached author embeddings for PAN15 processed CSVs.")
    parser.add_argument("--dataset-name", default="pan15_eng", help="Dataset prefix used in the embedding cache filename.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--aggregation", choices=["mean", "mean_std"], default="mean", help="How to aggregate document embeddings into one author vector.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cuda or cpu.")
    parser.add_argument("--train-csv", default=PROCESSED_DATA_DIR / "train.csv", type=Path)
    parser.add_argument("--test-csv", default=PROCESSED_DATA_DIR / "test.csv", type=Path)
    parser.add_argument("--output-dir", default=EMBEDDINGS_DIR, type=Path, help="Directory where the .npz cache and metadata will be saved.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate embeddings even if the target cache already exists.")
    return parser.parse_args()


def cache_name(model_name: str, aggregation: str, dataset_name: str = "pan15_eng") -> str:
    safe_model_name = model_name.replace("/", "__")
    safe_dataset_name = dataset_name.replace("/", "_").replace(" ", "_")
    return f"{safe_dataset_name}_{safe_model_name}_{aggregation}.npz"


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def split_documents(text: str) -> list[str]:
    text = str(text)

    if DOCUMENT_SEPARATOR in text:
        documents = [ document.strip() for document in text.split(DOCUMENT_SEPARATOR) if document.strip()]
        return documents

    documents = [ document.strip() for document in text.split(LEGACY_DOCUMENT_SEPARATOR) if document.strip()]
    return documents


def aggregate_embeddings(document_embeddings: np.ndarray, aggregation: str) -> np.ndarray:
    if document_embeddings.size == 0:
        raise ValueError("Cannot aggregate an author with no document embeddings.")

    mean_embedding = document_embeddings.mean(axis=0)

    if aggregation == "mean":
        return mean_embedding
    if aggregation == "mean_std":
        std_embedding = document_embeddings.std(axis=0)
        return np.concatenate([mean_embedding, std_embedding])

    raise ValueError(f"Unsupported aggregation: {aggregation}")


def encode_split(df: pd.DataFrame, model: object, aggregation: str, batch_size: int) -> np.ndarray:
    author_embeddings = []

    for row_number, row in enumerate(df.itertuples(index=False), start=1):
        documents = split_documents(row.text)
        if not documents:
            raise ValueError(f"Author {row.author_id} has no valid documents.")

        document_embeddings = model.encode(documents, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        author_embeddings.append(aggregate_embeddings(document_embeddings, aggregation))

        if row_number % 25 == 0 or row_number == len(df):
            print(f"Encoded {row_number}/{len(df)} authors")

    return np.vstack(author_embeddings).astype(np.float32)


def read_processed_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_columns = {
        "author_id",
        "gender",
        "age_group",
        "text"
    }
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    return df


def generate_embeddings(args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / cache_name(args.model_name, args.aggregation, args.dataset_name)
    metadata_path = output_path.with_suffix(".json")

    if output_path.exists() and not args.overwrite:
        print(f"Embedding cache already exists: {output_path}")
        print("Use --overwrite to regenerate it.")
        return output_path

    train_df = read_processed_csv(args.train_csv)
    test_df = read_processed_csv(args.test_csv)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name, device=args.device)

    print("Encoding train split")
    x_train = encode_split(train_df, model, args.aggregation, args.batch_size)

    print("Encoding test split")
    x_test = encode_split(test_df, model, args.aggregation, args.batch_size)

    np.savez_compressed(
        output_path,
        x_train=x_train,
        x_test=x_test,
        train_author_ids=train_df["author_id"].to_numpy(dtype=str),
        test_author_ids=test_df["author_id"].to_numpy(dtype=str),
        y_train_gender=train_df["gender"].to_numpy(dtype=str),
        y_test_gender=test_df["gender"].to_numpy(dtype=str),
        y_train_age=train_df["age_group"].to_numpy(dtype=str),
        y_test_age=test_df["age_group"].to_numpy(dtype=str),
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "aggregation": args.aggregation,
        "document_separator": DOCUMENT_SEPARATOR,
        "train_csv": str(args.train_csv),
        "test_csv": str(args.test_csv),
        "output_path": str(output_path),
        "x_train_shape": list(x_train.shape),
        "x_test_shape": list(x_test.shape),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved embeddings to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    return output_path


def main() -> None:
    args = parse_args()
    generate_embeddings(args)


if __name__ == "__main__":
    main()
