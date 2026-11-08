"""
Turn files on disk into plain-text SourceDocument objects.

WHY THIS FILE EXISTS
--------------------
Company knowledge arrives as PDFs, Word files, and Markdown. Everything after
this point (chunking, embedding, retrieval) wants one simple shape: text plus
metadata about where it came from. This module absorbs all file-format mess so
the rest of the pipeline never thinks about formats.

FOLDER CONVENTION
-----------------
The first sub-folder under the root is the department:
    data/raw/it/vpn_setup.md   -> department "it"
    data/raw/hr/leave_policy.pdf -> department "hr"
The department is stored with every chunk so retrieval can later be filtered
(e.g. only search HR documents for a leave question).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt", ".json"}
KNOWN_DEPARTMENTS = {"it", "hr"}


@dataclass(frozen=True)
class SourceDocument:
    """
    One unit of loaded text: a whole file, or one page of a PDF.

    WHY A DATACLASS: it's a plain bundle of values with no behaviour beyond
    building metadata. `frozen=True` makes instances immutable, so no later
    pipeline stage can accidentally modify the text we loaded.

    WHY PDFs ARE SPLIT PER PAGE: it lets answers cite a page number
    ("see page 4 of the Leave Policy"), which helps employees trust and verify.
    """

    text: str
    source: str            # path relative to the raw root, e.g. "hr/leave_policy.pdf"
    department: str        # "it", "hr", or "general"
    title: str             # human-friendly name, e.g. "Leave Policy"
    page: int | None = None  # 1-based page number for PDFs; None for other formats

    def metadata(self) -> dict:
        """
        Return the metadata dict stored next to every chunk of this document.

        WHY None VALUES ARE OMITTED: Chroma only accepts str/int/float/bool as
        metadata values and raises an error on None. Omitting `page` for
        non-PDF files avoids that.
        """
        meta = {"source": self.source, "department": self.department, "title": self.title}
        if self.page is not None:
            meta["page"] = self.page
        return meta


def _title_from_path(path: Path) -> str:
    """
    Make a readable title from a filename: "leave_policy-2025.pdf" -> "Leave Policy 2025".

    WHY: the title is shown in answers as the source and is also prepended to
    chunks before embedding (see ingest.py) to give retrieval extra context.
    """
    return path.stem.replace("_", " ").replace("-", " ").title()


def _department_for(path: Path, root: Path) -> str:
    """
    Infer the department from the first folder under `root`.

    Falls back to "general" for files placed directly in root or in an unknown
    folder, so a misplaced file is still indexed instead of dropped.
    """
    try:
        first = path.relative_to(root).parts[0].lower()
    except (ValueError, IndexError):
        return "general"
    return first if first in KNOWN_DEPARTMENTS else "general"


def _read_pdf(path: Path) -> list[tuple[str, int | None]]:
    """
    Extract text from each page of a PDF.

    Returns a list of (page_text, page_number). Pages with no extractable text
    come back as "" and are filtered later. Scanned PDFs (images of text) fall
    in this category: pypdf can't read them, and they need OCR, which is out of
    scope here.
    """
    reader = PdfReader(str(path))
    return [(page.extract_text() or "", n) for n, page in enumerate(reader.pages, start=1)]


def _read_docx(path: Path) -> list[tuple[str, int | None]]:
    """
    Extract text from a Word file as one document.

    - Heading paragraphs are prefixed with "# " so section titles survive and
      help the embedding understand what each passage is about.
    - Tables are flattened to "cell | cell | cell" lines and appended AFTER the
      paragraphs. Their original position in the document is lost, which is an
      acceptable trade-off for FAQ-style content.
    Word files have no reliable page concept, so page is None.
    """
    doc = DocxDocument(str(path))
    blocks: list[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "") if p.style is not None else ""
        blocks.append(f"# {text}" if style.startswith("Heading") else text)
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                blocks.append(" | ".join(cells))
    return [("\n\n".join(blocks), None)]


def _read_text(path: Path) -> list[tuple[str, int | None]]:
    """
    Read a Markdown or plain-text file as-is.

    Markdown is deliberately NOT converted to plain text: the "#" headings and
    list markers are useful signal for the embedding model, and they keep
    paragraph boundaries intact for the chunker.
    `errors="replace"` stops one stray bad byte from failing the whole file.
    """
    return [(path.read_text(encoding="utf-8", errors="replace"), None)]


def _read_json(path: Path) -> list[tuple[str, int | None]]:
    """Render JSON records as compact text while keeping related fields together."""
    data = json.loads(path.read_text(encoding="utf-8"))
    records: list[str] = []

    def visit(value, context: str = "") -> None:
        if isinstance(value, dict):
            fields = []
            nested = []
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    nested.append((key, child))
                else:
                    fields.append(f"{key.replace('_', ' ')}: {child}")
            if fields:
                prefix = f"{context}: " if context else ""
                records.append(prefix + " | ".join(fields))
            for key, child in nested:
                label = key.replace("_", " ")
                visit(child, f"{context} {label}".strip())
        elif isinstance(value, list):
            for item in value:
                visit(item, context)
        else:
            prefix = f"{context}: " if context else ""
            records.append(f"{prefix}{value}")

    visit(data)
    return [("\n\n".join(records), None)]


_READERS = {
    ".pdf": _read_pdf,
    ".docx": _read_docx,
    ".md": _read_text,
    ".txt": _read_text,
    ".json": _read_json,
}


def load_file(path: Path, root: Path) -> list[SourceDocument]:
    """
    Load one file into zero or more SourceDocuments.

    Args:
        path: the file to read.
        root: the raw-data root, used to compute the relative `source` and
            the department.

    Returns:
        One document per PDF page, or one per other file. Empty list if the
        format is unsupported, the file has no extractable text, or reading failed.

    WHY ERRORS ARE LOGGED, NOT RAISED: with hundreds of documents, one corrupt
    or password-protected PDF should not abort the whole ingest. The skipped
    file is logged so you can fix it, and everything else still gets indexed.
    """
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        logger.warning("Skipping unsupported file type: %s", path)
        return []

    try:
        parts = reader(path)
    except Exception as exc:
        logger.warning("Could not read %s (%s); skipping", path, exc)
        return []

    source = path.relative_to(root).as_posix()
    department = _department_for(path, root)
    title = _title_from_path(path)
    docs = [
        SourceDocument(text=text.strip(), source=source, department=department,
                       title=title, page=page)
        for text, page in parts
        if text.strip()
    ]
    if not docs:
        logger.warning("No extractable text in %s (scanned PDF or empty file?)", path)
    return docs


def load_directory(root: str | Path) -> list[SourceDocument]:
    """
    Recursively load every supported file under `root`.

    Files are visited in sorted order so repeated runs produce the same
    document order (and therefore the same chunk IDs), which makes ingestion
    reproducible and easier to debug.

    Raises:
        FileNotFoundError: if `root` doesn't exist. A wrong path should be a
        loud error, not a silent "0 documents indexed".
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Raw documents folder not found: {root}")

    docs: list[SourceDocument] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name.startswith("."):  # .DS_Store, .gitkeep, etc.
            continue
        docs.extend(load_file(path, root))
    return docs


def main() -> None:
    """Smoke-test JSON loading against the repository's sample datasets."""
    root = Path("smart_desk_modules/data_sets")
    json_paths = sorted(root.rglob("*.json"))
    docs = [doc for path in json_paths for doc in load_file(path, root)]

    print(f"Loaded {len(docs)} of {len(json_paths)} JSON files from {root}")
    for doc in docs:
        preview = doc.text[:160].replace("\n", " ")
        print(f"{doc.source}: {preview}")


if __name__ == "__main__":
    main()