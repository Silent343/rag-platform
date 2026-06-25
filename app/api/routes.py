"""
REST API routes for the RAG service.

Endpoints (REST conventions on the /documents and /query resources):
- POST   /api/documents        — ingest a document (multipart upload)
- GET    /api/documents        — list ingested documents
- DELETE /api/documents/{id}   — remove a document
- POST   /api/query            — ask a question
"""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_rag_service
from app.models.schemas import (
    DocumentSummary,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.rag_service import RagService

router = APIRouter(prefix="/api", tags=["rag"])


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(
    file: UploadFile = File(...),
    service: RagService = Depends(get_rag_service),
) -> IngestResponse:
    """
    Ingests an uploaded document into the knowledge base.

    Args:
        file: The uploaded file (PDF, TXT or MD).
        service: The injected RAG service.

    Returns:
        Details of the ingested document.

    Raises:
        HTTPException: 400 if the file cannot be processed.
    """
    content = await file.read()
    try:
        result = service.ingest(file.filename or "untitled", content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return IngestResponse(**result)


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    service: RagService = Depends(get_rag_service),
) -> list[DocumentSummary]:
    """
    Lists the documents currently stored in the knowledge base.

    Args:
        service: The injected RAG service.

    Returns:
        A summary of each stored document.
    """
    docs = service._store.list_documents()
    return [DocumentSummary(**doc) for doc in docs]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    service: RagService = Depends(get_rag_service),
) -> None:
    """
    Deletes a document and all of its chunks.

    Args:
        document_id: The id of the document to delete.
        service: The injected RAG service.
    """
    service._store.delete_document(document_id)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """
    Answers a question grounded in the stored documents.

    Args:
        request: The question payload.
        service: The injected RAG service.

    Returns:
        The generated answer plus the source chunks that grounded it.
    """
    result = service.answer(request.question)
    return QueryResponse(**result)
