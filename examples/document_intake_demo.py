"""
Example script demonstrating Module 3.19 concepts:
- Multi-format document loading (.txt, .md, .html)
- Preserving source identity and region metadata
- Gracefully handling missing/corrupt files without crashing
- Inspecting length and sample content
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.document_loader import DocumentLoader, LoadedDocument


def run_document_intake_demonstration():
    print("=" * 60)
    print("1. MULTI-FORMAT DOCUMENT INTAKE DEMONSTRATION")
    print("=" * 60)

    loader = DocumentLoader(default_region="India")

    # Create temporary directory with mock HR policy files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Create TXT document
        txt_file = temp_path / "India_Leave_Policy.txt"
        txt_file.write_text(
            "India Leave Policy Section 4: Employees in India are entitled to 20 paid leave days annually.",
            encoding="utf-8"
        )

        # 2. Create Markdown document
        md_file = temp_path / "Global_Employee_Handbook.md"
        md_file.write_text(
            "# Global Employee Handbook\n\n## Section 1: Code of Conduct\nAll employees must maintain professional integrity.",
            encoding="utf-8"
        )

        # 3. Create HTML document
        html_file = temp_path / "US_Benefits_Guide.html"
        html_file.write_text(
            "<html><body><h1>US Benefits Guide</h1><p>Health insurance covers eligible full-time employees.</p></body></html>",
            encoding="utf-8"
        )

        # 4. Create an unreadable / unsupported dummy file to test error survival
        corrupt_file = temp_path / "Corrupt_File.unsupported"
        corrupt_file.write_text("Binary or unknown content", encoding="utf-8")

        print(f"\nIngesting documents from temporary path: {temp_path}\n")

        # Ingest directory
        documents = loader.load_directory(temp_path, region="India")

        # Log individual intake results
        for doc in documents:
            if doc.status == "OK":
                print(f"[OK] {doc.source:<30} | {doc.char_count:>4} chars | Region: {doc.region:<6} | Preview: {doc.preview(50)!r}")
            else:
                print(f"[SKIP] {doc.source:<28} | ERROR: {doc.error}")

        # Display Intake Summary
        print("\n" + "=" * 60)
        print("2. DOCUMENT INTAKE SUMMARY")
        print("=" * 60)
        summary = loader.summarize_intake(documents)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_document_intake_demonstration()
