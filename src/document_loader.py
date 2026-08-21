"""
HRAssist - Document Loading & Multi-Format Intake Pipeline
Module 3.19: Document Loading & Multi-Format Intake

This module provides tools for:
1. Intake of multi-format documents (.pdf, .txt, .md, .html, .htm).
2. Converting diverse file formats into unified plain-text representations.
3. Attaching source and regional metadata to every loaded document.
4. Surviving corrupted/unreadable files without crashing the intake pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import List, Dict, Any, Optional

# Optional third-party imports with fallback handling
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass
class LoadedDocument:
    source: str
    file_path: str
    text: str
    extension: str
    region: str = "Global"
    char_count: int = 0
    status: str = "OK"  # "OK" or "FAILED"
    error: Optional[str] = None

    def __post_init__(self):
        if self.text:
            self.char_count = len(self.text)

    def preview(self, max_chars: int = 80) -> str:
        clean_text = self.text.replace("\n", " ").strip()
        if len(clean_text) > max_chars:
            return clean_text[:max_chars] + "..."
        return clean_text


class DocumentLoader:
    """
    Multi-format document loader with robust exception handling and metadata tracking.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}

    def __init__(self, default_region: str = "Global"):
        self.default_region = default_region

    def extract_text_from_file(self, path: Path) -> str:
        """
        Extracts plain text content from a given file path based on its file extension.
        """
        suffix = path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {suffix}")

        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8", errors="ignore")

        elif suffix in (".html", ".htm"):
            content = path.read_text(encoding="utf-8", errors="ignore")
            if HAS_BS4:
                return BeautifulSoup(content, "html.parser").get_text(" ")
            else:
                # Fallback basic tag stripper if bs4 is not installed
                clean = re.sub(r"<[^>]+>", " ", content)
                return " ".join(clean.split())

        elif suffix == ".pdf":
            if not HAS_PYPDF:
                raise RuntimeError("pypdf package is required to extract PDF text.")
            reader = PdfReader(str(path))
            extracted_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text)
            return "\n".join(extracted_pages)

        raise ValueError(f"Unhandled file format: {suffix}")

    def load_single_file(self, path: Path, region: Optional[str] = None) -> LoadedDocument:
        """
        Loads a single document file gracefully. Captures errors without throwing exceptions.
        """
        assigned_region = region or self.default_region
        source_name = path.name
        file_ext = path.suffix.lower()

        if not path.is_file():
            return LoadedDocument(
                source=source_name,
                file_path=str(path),
                text="",
                extension=file_ext,
                region=assigned_region,
                status="FAILED",
                error="File does not exist or is not a regular file."
            )

        try:
            text = self.extract_text_from_file(path)
            return LoadedDocument(
                source=source_name,
                file_path=str(path),
                text=text,
                extension=file_ext,
                region=assigned_region,
                status="OK",
                error=None
            )
        except Exception as e:
            return LoadedDocument(
                source=source_name,
                file_path=str(path),
                text="",
                extension=file_ext,
                region=assigned_region,
                status="FAILED",
                error=str(e)
            )

    def load_directory(
        self,
        directory_path: Path,
        region: Optional[str] = None,
        recursive: bool = True
    ) -> List[LoadedDocument]:
        """
        Scans and ingests all supported files in a directory.
        """
        documents = []
        if not directory_path.exists() or not directory_path.is_dir():
            return documents

        pattern = "**/*" if recursive else "*"
        for path in directory_path.glob(pattern):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                doc = self.load_single_file(path, region=region)
                documents.append(doc)

        return documents

    @staticmethod
    def summarize_intake(documents: List[LoadedDocument]) -> Dict[str, Any]:
        """
        Computes summary statistics for ingested documents.
        """
        total = len(documents)
        successful = [d for d in documents if d.status == "OK"]
        failed = [d for d in documents if d.status == "FAILED"]

        format_counts = {}
        for d in documents:
            format_counts[d.extension] = format_counts.get(d.extension, 0) + 1

        total_chars = sum(d.char_count for d in successful)

        return {
            "total_documents": total,
            "successful": len(successful),
            "failed": len(failed),
            "total_character_count": total_chars,
            "format_breakdown": format_counts,
            "failed_details": [{"source": d.source, "error": d.error} for d in failed]
        }
