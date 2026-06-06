from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "eng"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DOCUMENT_SEPARATOR = " <DOC_SEP> "

TRUTH_COLUMNS = ["author_id", "gender", "age_group", "extroverted", "stable", "agreeable", "conscientious", "open"]
OUTPUT_COLUMNS = ["author_id", "gender", "age_group", "extroverted", "stable", "agreeable", "conscientious", "open", "lang", "n_docs_raw", "n_docs", "n_duplicates_removed", "text"]

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
INITIAL_RT_RE = re.compile(r"^\s*RT\b:?", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text or ""
    text = URL_RE.sub("<URL>", text)
    text = MENTION_RE.sub("<USER>", text)
    text = INITIAL_RT_RE.sub("RT", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def read_truth(split_dir: Path) -> dict[str, dict[str, str]]:
    truth_path = split_dir / "truth.txt"
    labels = {}

    with truth_path.open("r", encoding="utf-8", newline="") as truth_file:
        for line_number, line in enumerate(truth_file, start=1):
            line = line.strip()
            if not line:
                continue

            values = line.split(":::")
            if len(values) != len(TRUTH_COLUMNS):
                raise ValueError(
                    f"Invalid truth row in {truth_path} at line {line_number}: "
                    f"expected {len(TRUTH_COLUMNS)} columns, found {len(values)}"
                )

            row = dict(zip(TRUTH_COLUMNS, values))
            labels[row["author_id"]] = row

    return labels


def read_author_xml(xml_path: Path) -> tuple[str, str, list[str]]:
    root = ET.parse(xml_path).getroot()
    author_id = root.attrib.get("id", xml_path.stem)
    lang = root.attrib.get("lang", "")
    documents = [
        normalize_text(document.text or "")
        for document in root.findall(".//document")
    ]
    return author_id, lang, documents


def remove_exact_duplicates(documents: list[str]) -> tuple[list[str], int]:
    seen = set()
    unique_documents = []

    for document in documents:
        if document in seen:
            continue
        seen.add(document)
        unique_documents.append(document)

    return unique_documents, len(documents) - len(unique_documents)


def build_split(split: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    split_dir = RAW_DATA_DIR / split
    labels_by_author = read_truth(split_dir)
    rows = []
    total_raw_documents = 0
    total_documents = 0
    total_duplicates_removed = 0

    for xml_path in sorted(split_dir.glob("*.xml")):
        author_id, lang, documents = read_author_xml(xml_path)

        if author_id not in labels_by_author:
            raise ValueError(f"Missing truth label for {author_id} in split {split}")

        unique_documents, duplicates_removed = remove_exact_duplicates(documents)
        label_row = labels_by_author[author_id]

        rows.append({
                **label_row,
                "lang": lang,
                "n_docs_raw": str(len(documents)),
                "n_docs": str(len(unique_documents)),
                "n_duplicates_removed": str(duplicates_removed),
                "text": DOCUMENT_SEPARATOR.join(unique_documents),
        })

        total_raw_documents += len(documents)
        total_documents += len(unique_documents)
        total_duplicates_removed += duplicates_removed

    xml_author_ids = {row["author_id"] for row in rows}
    missing_xml = sorted(set(labels_by_author) - xml_author_ids)
    if missing_xml:
        raise ValueError(f"Missing XML files in split {split}: {missing_xml}")

    stats = {
        "authors": len(rows),
        "raw_documents": total_raw_documents,
        "documents": total_documents,
        "duplicates_removed": total_duplicates_removed,
    }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for split in ("train", "test"):
        rows, stats = build_split(split)
        output_path = PROCESSED_DATA_DIR / f"{split}.csv"
        write_csv(output_path, rows)
        print(
            f"{split}: {stats['authors']} authors, "
            f"{stats['raw_documents']} raw documents, "
            f"{stats['documents']} processed documents, "
            f"{stats['duplicates_removed']} duplicates removed -> {output_path}"
        )


if __name__ == "__main__":
    main()
