"""End-to-end corpus ingestion with explicit failure accounting."""

from pathlib import Path
from typing import List, Tuple

from src.document_chunker import tag_chunks
from src.document_loader import DocumentLoader
from src.text_cleaner import clean_text
from src.token_chunker import token_chunks


def ingest(
    folder: str | Path,
    chunk_size: int = 100,
    overlap: int = 20,
    region: str = "Global",
) -> Tuple[List[Path], int, list, List[Tuple[str, str]]]:
    """Load, clean, chunk, and tag every regular file under ``folder``.

    Returns ``(files, documents_ingested, chunks, failures)``. Unsupported,
    unreadable, and otherwise invalid files are included in ``failures`` so
    that ``documents_ingested + len(failures) == len(files)`` can be checked.
    """

    folder_path = Path(folder)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Ingestion folder does not exist: {folder_path}")

    files = sorted(path for path in folder_path.rglob("*") if path.is_file())
    loader = DocumentLoader(default_region=region)
    chunks = []
    documents_ingested = 0
    failures = []

    for path in files:
        try:
            document = loader.load_single_file(path, region=region)
            if document.status == "FAILED":
                raise RuntimeError(document.error or "Document loading failed")

            cleaned = clean_text(document.text)
            raw_chunks = token_chunks(cleaned, chunk_size=chunk_size, overlap=overlap)
            chunks.extend(
                tag_chunks(
                    source=document.source,
                    chunks=raw_chunks,
                    region=document.region,
                )
            )
            documents_ingested += 1
        except Exception as error:
            failures.append((path.name, str(error)))

    return files, documents_ingested, chunks, failures


def main() -> None:
    """Run ingestion for the repository's sample data directory."""

    files, documents, chunks, failures = ingest(Path(__file__).parents[1] / "data")
    print(
        f"files={len(files)} docs={documents} "
        f"chunks={len(chunks)} failures={len(failures)}"
    )
    assert documents + len(failures) == len(files), "a document was silently dropped!"

    for name, error in failures:
        print(f"FAILED: {name} {error}")

    if chunks:
        print(f"sample: {chunks[0]['text'][:80]} | {chunks[0]['metadata']}")


if __name__ == "__main__":
    main()