"""
Document loading: extracts plain text from uploaded files.

Supports PDF (via pypdf) and plain-text files. The extracted text is handed to
the chunker downstream.
"""

from io import BytesIO

from pypdf import PdfReader


class DocumentLoader:
    """Extracts text content from raw uploaded file bytes."""

    @staticmethod
    def extract_text(filename: str, content: bytes) -> str:
        """
        Extracts text from a file based on its extension.

        Args:
            filename: The original file name (used to detect the type).
            content: The raw file bytes.

        Returns:
            The extracted text.

        Raises:
            ValueError: If the file type is unsupported or the file is empty.
        """
        if not content:
            raise ValueError("The uploaded file is empty.")

        lowered = filename.lower()
        if lowered.endswith(".pdf"):
            return DocumentLoader._extract_pdf(content)
        if lowered.endswith((".txt", ".md")):
            return content.decode("utf-8", errors="ignore")

        raise ValueError(
            f"Unsupported file type: {filename}. Allowed: .pdf, .txt, .md."
        )

    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        """
        Extracts and concatenates the text of every page in a PDF.

        Args:
            content: The raw PDF bytes.

        Returns:
            The full document text, with pages separated by blank lines.
        """
        reader = PdfReader(BytesIO(content))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
