"""Pydantic schemas shared by the REST API and frontend contract."""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    already_indexed: bool = False


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer.")


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    page_number: int | None = None
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunks: int
