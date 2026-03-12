import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    path = parsed.path.rstrip("/")
    normalized = parsed._replace(path=path, fragment="")
    return urlunsplit(normalized)


def has_embedding_for_url(embeddings_file: str, target_url: str) -> bool:
    normalized_target = normalize_url(target_url)

    with open(embeddings_file, "r", encoding="utf-8") as file_handle:
        records = json.load(file_handle)

    for record in records:
        metadata = record.get("metadata", {}) or {}
        source = metadata.get("source") or metadata.get("source_url") or ""

        if not source:
            continue

        if normalize_url(source) == normalized_target:
            return True

    return False


def find_embeddings_across_library(embeddings_dir: str, target_url: str) -> bool:
    directory = Path(embeddings_dir)
    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")

    for json_file in sorted(directory.glob("*.json")):
        if has_embedding_for_url(str(json_file), target_url):
            return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a URL already has embeddings")
    parser.add_argument("--url", required=True, help="URL to check")
    parser.add_argument(
        "--embeddings-dir",
        default="embedding_library",
        help="Directory containing embeddings JSON files",
    )

    args = parser.parse_args()

    url_exists = find_embeddings_across_library(args.embeddings_dir, args.url)

    if url_exists:
        print(f"FOUND: {args.url}")
        print("True")
    else:
        print(f"NOT FOUND: {args.url}")
        print("False")


if __name__ == "__main__":
    main()
