"""REST API routes for documents and grounded RAG queries."""

from functools import lru_cache

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.dependencies import get_rag_service
from app.config import Settings, get_settings
from app.errors import ProviderError
from app.middleware.rate_limit import RateLimiter
from app.models.schemas import DocumentSummary, IngestResponse, QueryRequest, QueryResponse
from app.services.rag_service import RagService

router = APIRouter(prefix="/api", tags=["rag"])


@lru_cache
def get_rate_limiter() -> RateLimiter:
    settings = get_settings()
    return RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)


def enforce_rate_limit(request: Request, limiter: RateLimiter = Depends(get_rate_limiter)) -> None:
    limiter.check(request)


def provider_error(exc: ProviderError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@router.post("/documents", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, dependencies=[Depends(enforce_rate_limit)])
async def ingest_document(file: UploadFile = File(...), service: RagService = Depends(get_rag_service), settings: Settings = Depends(get_settings)) -> IngestResponse:
    """Ingests a PDF, TXT or Markdown document after enforcing size limits."""
    filename = file.filename or "untitled"
    if not filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Allowed file types: .pdf, .txt, .md.")
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        max_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=f"Files must be at most {max_mb} MB.")
    try:
        result = service.ingest(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ProviderError as exc:
        raise provider_error(exc) from exc
    return IngestResponse(**result)


@router.get("/documents", response_model=list[DocumentSummary], dependencies=[Depends(enforce_rate_limit)])
async def list_documents(service: RagService = Depends(get_rag_service)) -> list[DocumentSummary]:
    return [DocumentSummary(**doc) for doc in service._store.list_documents()]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(enforce_rate_limit)])
async def delete_document(document_id: str, service: RagService = Depends(get_rag_service)) -> None:
    service._store.delete_document(document_id)


@router.post("/query", response_model=QueryResponse, dependencies=[Depends(enforce_rate_limit)])
async def query(request: QueryRequest, service: RagService = Depends(get_rag_service)) -> QueryResponse:
    try:
        return QueryResponse(**service.answer(request.question))
    except ProviderError as exc:
        raise provider_error(exc) from exc
