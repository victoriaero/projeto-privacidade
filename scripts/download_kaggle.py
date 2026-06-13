from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "blog_authorship"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "blog_authorship"

KAGGLE_DATASET = "rtatman/blog-authorship-corpus"

# Use um separador improvável de aparecer naturalmente no texto.
# Isso é mais seguro do que juntar posts com dois espaços.
DOCUMENT_SEPARATOR = "\n<DOC_SEP>\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and prepare the Kaggle Blog Authorship Corpus into the "
            "author-level CSV format expected by the privacy attack pipeline."
        )
    )

    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument(
        "--min-posts-per-author",
        type=int,
        default=1,
        help="Drop authors with fewer than this number of posts.",
    )

    parser.add_argument(
        "--max-posts-per-author",
        type=int,
        default=None,
        help=(
            "Optional cap on posts per author. Useful for quick experiments "
            "on the large corpus."
        ),
    )

    parser.add_argument(
        "--max-authors",
        type=int,
        default=None,
        help=(
            "Optional cap on number of authors after filtering. Useful for "
            "testing the pipeline before running the full dataset."
        ),
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the Kaggle dataset again even if files already exist.",
    )

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def download_dataset(dataset: str, raw_dir: Path, force_download: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(raw_dir.glob("*"))
    if existing_files and not force_download:
        print(f"Raw data already exists in {raw_dir}. Skipping download.")
        return

    if force_download and raw_dir.exists():
        shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "blog-authorship-corpus.zip"

    run_command(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset,
            "-p",
            str(raw_dir),
            "--force",
        ]
    )

    zip_files = list(raw_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(
            f"No zip file was downloaded to {raw_dir}. "
            "Check whether Kaggle credentials are configured."
        )

    # Se o nome do zip vier diferente, pega o primeiro.
    zip_path = zip_files[0]

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(raw_dir)


def normalize_gender(value: object) -> str:
    gender = str(value).strip().lower()

    if gender in {"m", "male"}:
        return "male"

    if gender in {"f", "female"}:
        return "female"

    return gender


def age_to_group(value: object) -> str:
    age = int(value)

    if 13 <= age <= 19:
        return "teens"

    if 20 <= age <= 29:
        return "twenties"

    if age >= 30:
        return "30+"

    return "other"


def clean_text(value: object) -> str:
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_blogtext_csv(raw_dir: Path) -> Path | None:
    candidates = list(raw_dir.rglob("blogtext.csv"))

    if candidates:
        return candidates[0]

    csv_files = list(raw_dir.rglob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    return None


def prepare_from_csv(
    csv_path: Path,
    min_posts_per_author: int,
    max_posts_per_author: int | None,
) -> pd.DataFrame:
    print(f"Preparing from CSV: {csv_path}")

    author_posts: dict[str, list[str]] = defaultdict(list)
    author_gender: dict[str, str] = {}
    author_age_group: dict[str, str] = {}

    chunks = pd.read_csv(csv_path, chunksize=50_000)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.columns = [column.strip().lower() for column in chunk.columns]

        required = {"id", "gender", "age", "text"}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns {sorted(missing)}. "
                f"Available columns: {list(chunk.columns)}"
            )

        for row in chunk.itertuples(index=False):
            author_id = str(getattr(row, "id"))
            gender = normalize_gender(getattr(row, "gender"))
            age_group = age_to_group(getattr(row, "age"))
            text = clean_text(getattr(row, "text"))

            if not text:
                continue

            if (
                max_posts_per_author is not None
                and len(author_posts[author_id]) >= max_posts_per_author
            ):
                continue

            author_posts[author_id].append(text)
            author_gender[author_id] = gender
            author_age_group[author_id] = age_group

        print(f"Processed chunk {chunk_number}")

    return build_author_dataframe(
        author_posts=author_posts,
        author_gender=author_gender,
        author_age_group=author_age_group,
        min_posts_per_author=min_posts_per_author,
    )


def parse_metadata_from_filename(path: Path) -> tuple[str, str, str] | None:
    # Formato comum do corpus original:
    # 5114.male.25.indUnk.Scorpio.xml
    parts = path.stem.split(".")

    if len(parts) < 3:
        return None

    author_id = parts[0]
    gender = normalize_gender(parts[1])
    age_group = age_to_group(parts[2])

    return author_id, gender, age_group


def extract_posts_from_xml_text(xml_text: str) -> list[str]:
    posts = re.findall(r"<post>(.*?)</post>", xml_text, flags=re.DOTALL | re.IGNORECASE)
    return [clean_text(post) for post in posts if clean_text(post)]


def prepare_from_xml_files(
    raw_dir: Path,
    min_posts_per_author: int,
    max_posts_per_author: int | None,
) -> pd.DataFrame:
    print(f"Preparing from XML files in: {raw_dir}")

    author_posts: dict[str, list[str]] = defaultdict(list)
    author_gender: dict[str, str] = {}
    author_age_group: dict[str, str] = {}

    xml_files = list(raw_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(
            f"No blogtext.csv or XML files found under {raw_dir}."
        )

    for file_number, path in enumerate(xml_files, start=1):
        metadata = parse_metadata_from_filename(path)
        if metadata is None:
            continue

        author_id, gender, age_group = metadata

        xml_text = path.read_text(encoding="utf-8", errors="ignore")
        posts = extract_posts_from_xml_text(xml_text)

        if max_posts_per_author is not None:
            posts = posts[:max_posts_per_author]

        if not posts:
            continue

        author_posts[author_id].extend(posts)
        author_gender[author_id] = gender
        author_age_group[author_id] = age_group

        if file_number % 500 == 0 or file_number == len(xml_files):
            print(f"Processed XML file {file_number}/{len(xml_files)}")

    return build_author_dataframe(
        author_posts=author_posts,
        author_gender=author_gender,
        author_age_group=author_age_group,
        min_posts_per_author=min_posts_per_author,
    )


def build_author_dataframe(
    author_posts: dict[str, list[str]],
    author_gender: dict[str, str],
    author_age_group: dict[str, str],
    min_posts_per_author: int,
) -> pd.DataFrame:
    rows = []

    for author_id, posts in author_posts.items():
        if len(posts) < min_posts_per_author:
            continue

        rows.append(
            {
                "author_id": author_id,
                "gender": author_gender[author_id],
                "age_group": author_age_group[author_id],
                "text": DOCUMENT_SEPARATOR.join(posts),
                "n_posts": len(posts),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No authors remained after filtering.")

    df = df.sort_values("author_id").reset_index(drop=True)

    print("Prepared author-level dataset")
    print(f"Authors: {len(df)}")
    print("Gender distribution:")
    print(df["gender"].value_counts())
    print("Age group distribution:")
    print(df["age_group"].value_counts())
    print("Posts per author:")
    print(df["n_posts"].describe())

    return df


def stratification_label(df: pd.DataFrame) -> pd.Series:
    return df["gender"].astype(str) + "__" + df["age_group"].astype(str)


def split_train_test(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify = stratification_label(df)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    return (
        train_df.sort_values("author_id").reset_index(drop=True),
        test_df.sort_values("author_id").reset_index(drop=True),
    )


def save_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_columns = ["author_id", "gender", "age_group", "text"]

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    metadata_path = processed_dir / "metadata.json"

    train_df[output_columns].to_csv(train_path, index=False)
    test_df[output_columns].to_csv(test_path, index=False)

    metadata = {
        "dataset": "blog_authorship_corpus",
        "document_separator": DOCUMENT_SEPARATOR,
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train_authors": int(len(train_df)),
        "n_test_authors": int(len(test_df)),
        "train_gender_distribution": train_df["gender"].value_counts().to_dict(),
        "test_gender_distribution": test_df["gender"].value_counts().to_dict(),
        "train_age_distribution": train_df["age_group"].value_counts().to_dict(),
        "test_age_distribution": test_df["age_group"].value_counts().to_dict(),
        "train_posts_per_author": train_df["n_posts"].describe().to_dict(),
        "test_posts_per_author": test_df["n_posts"].describe().to_dict(),
    }

    metadata_path.write_text(
        pd.Series(metadata).to_json(indent=2),
        encoding="utf-8",
    )

    print(f"Saved train CSV to {train_path}")
    print(f"Saved test CSV to {test_path}")
    print(f"Saved metadata to {metadata_path}")


def main() -> None:
    args = parse_args()

    download_dataset(
        dataset=args.dataset,
        raw_dir=args.raw_dir,
        force_download=args.force_download,
    )

    csv_path = find_blogtext_csv(args.raw_dir)

    if csv_path is not None:
        df = prepare_from_csv(
            csv_path=csv_path,
            min_posts_per_author=args.min_posts_per_author,
            max_posts_per_author=args.max_posts_per_author,
        )
    else:
        df = prepare_from_xml_files(
            raw_dir=args.raw_dir,
            min_posts_per_author=args.min_posts_per_author,
            max_posts_per_author=args.max_posts_per_author,
        )

    if args.max_authors is not None:
        df = (
            df.sample(
                n=min(args.max_authors, len(df)),
                random_state=args.seed,
            )
            .sort_values("author_id")
            .reset_index(drop=True)
        )
        print(f"Using sampled subset with {len(df)} authors")

    train_df, test_df = split_train_test(
        df=df,
        test_size=args.test_size,
        seed=args.seed,
    )

    save_outputs(
        train_df=train_df,
        test_df=test_df,
        processed_dir=args.processed_dir,
    )


if __name__ == "__main__":
    main()