"""Build the local knowledge index from source documents."""

import argparse

from smartdesk.rag.ingest import ingest


def main() -> None:
    """Load documents, create embeddings, and save them in the vector store."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="directory containing source documents (default: data/raw)",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="add or update documents without clearing the existing index",
    )
    args = parser.parse_args()

    report = ingest(raw_dir=args.raw_dir, reset=not args.no_reset)
    print(f"Ingestion complete: {report.documents} documents, {report.chunks} chunks")
    for department, count in sorted(report.by_department.items()):
        print(f"  {department}: {count} chunks")


if __name__ == "__main__":
    main()
