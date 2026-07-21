"""Regression tests for deterministic RAG utilities."""

import unittest

from fastapi import HTTPException
from starlette.requests import Request

from app.middleware.rate_limit import RateLimiter
from app.services.chunker import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.rag_service import RagService


class RAGUtilityTests(unittest.TestCase):
    def test_text_loader_labels_plain_text_as_page_one(self) -> None:
        pages = DocumentLoader(max_pdf_pages=100).extract_pages("notes.txt", b"First line\nSecond line")
        self.assertEqual(pages, [(1, "First line\nSecond line")])

    def test_chunker_preserves_all_words(self) -> None:
        chunks = TextChunker(chunk_size=16, chunk_overlap=4).split("one two three four five six")
        self.assertGreater(len(chunks), 1)
        self.assertIn("one", chunks[0])
        self.assertIn("six", chunks[-1])

    def test_context_identifies_page_number(self) -> None:
        context = RagService._build_context([{
            "filename": "guide.pdf", "page_number": 3, "text": "Important detail"
        }])
        self.assertIn("guide.pdf, page 3", context)

    def test_rate_limiter_rejects_excess_requests(self) -> None:
        request = Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)})
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check(request)
        with self.assertRaises(HTTPException) as raised:
            limiter.check(request)
        self.assertEqual(raised.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
