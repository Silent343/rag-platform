"""
Text chunking: splits a long document into overlapping chunks.

Chunking matters for RAG quality: chunks that are too big dilute relevance,
too small lose context. Overlap keeps sentences that straddle a boundary from
being cut off from their context.
"""


class TextChunker:
    """Splits text into fixed-size, overlapping chunks on word boundaries."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Args:
            chunk_size: Target chunk length in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.

        Raises:
            ValueError: If overlap is not smaller than the chunk size.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        """
        Splits text into overlapping chunks without cutting words in half.

        The algorithm walks the text in windows of ``chunk_size`` characters,
        then backs up to the last whitespace so a word is never split, and
        advances by ``chunk_size - chunk_overlap`` for the next window.

        Args:
            text: The full document text.

        Returns:
            A list of non-empty text chunks.
        """
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        length = len(normalized)

        while start < length:
            end = min(start + self._chunk_size, length)

            # Back up to the last space so we don't split a word (unless we're
            # at the very end of the text).
            if end < length:
                last_space = normalized.rfind(" ", start, end)
                if last_space > start:
                    end = last_space

            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= length:
                break
            start = end - self._chunk_overlap

        return chunks
