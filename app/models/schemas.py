"""
Pydantic schemas describing the request and response shapes of the API.

These are the contract between the FastAPI backend and the Angular frontend.
"""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """
    Returned after a document is ingested.

    Attributes:
        document_id: Stable id assigned to the ingested document.
        filename: Original file name.
        chunks_created: How many chunks the document was split into.
    """

    document_id: str
    filename: str
    chunks_created: int


class QueryRequest(BaseModel):
    """
    Body of a question sent to the RAG endpoint.

    Attributes:
        question: The user's natural-language question.
    """

    question: str = Field(..., min_length=1, description="The question to answer.")


class SourceChunk(BaseModel):
    """
    A single retrieved chunk used to support the answer.

    Attributes:
        document_id: Id of the source document.
        filename: Name of the source document.
        chunk_index: Position of the chunk within its document.
        text: The chunk's text content.
        score: Similarity score against the question (higher = closer).
    """

    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    """
    The answer to a question plus the sources that grounded it.

    Attributes:
        answer: The generated answer.
        sources: The chunks the model was given as context.
    """

    answer: str
    sources: list[SourceChunk]


class DocumentSummary(BaseModel):
    """
    Lightweight description of an ingested document for listing.

    Attributes:
        document_id: Stable document id.
        filename: Original file name.
        chunks: Number of chunks stored for this document.
    """

    document_id: str
    filename: str
    chunks: int
