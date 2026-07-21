"""Document loading with page-aware PDF extraction."""

from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class DocumentLoader:
    """Extracts page-labelled text content from uploaded files."""

    def __init__(self, max_pdf_pages: int) -> None:
        self._max_pdf_pages = max_pdf_pages

    def extract_pages(self, filename: str, content: bytes) -> list[tuple[int, str]]:
        if not content:
            raise ValueError("The uploaded file is empty.")
        lowered = filename.lower()
        if lowered.endswith(".pdf"):
            return self._extract_pdf_pages(content)
        if lowered.endswith((".txt", ".md")):
            text = content.decode("utf-8", errors="ignore").strip()
            if not text:
                raise ValueError("No text could be extracted from the document.")
            return [(1, text)]
        raise ValueError(f"Unsupported file type: {filename}. Allowed: .pdf, .txt, .md.")

    def extract_text(self, filename: str, content: bytes) -> str:
        return "\n\n".join(text for _, text in self.extract_pages(filename, content))

    def _extract_pdf_pages(self, content: bytes) -> list[tuple[int, str]]:
        try:
            reader = PdfReader(BytesIO(content))
        except PdfReadError as exc:
            raise ValueError("The PDF could not be read. It may be damaged or encrypted.") from exc
        if len(reader.pages) > self._max_pdf_pages:
            raise ValueError(f"PDFs can contain at most {self._max_pdf_pages} pages.")
        pages = [
            (number, text)
            for number, page in enumerate(reader.pages, start=1)
            if (text := (page.extract_text() or "").strip())
        ]
        if not pages:
            raise ValueError("No text could be extracted from the PDF.")
        return pages
