"""Persistent ChromaDB storage for page-aware document chunks."""

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """Stores and queries document embeddings in a ChromaDB collection."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir, settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, document_id: str, filename: str, content_hash: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self._collection.add(
            ids=[f"{document_id}::chunk::{index}" for index in range(len(chunks))],
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {"document_id": document_id, "filename": filename, "content_hash": content_hash, "chunk_index": index, "page_number": chunk["page_number"]}
                for index, chunk in enumerate(chunks)
            ],
        )

    def find_document_by_hash(self, content_hash: str) -> dict | None:
        result = self._collection.get(where={"content_hash": content_hash}, include=["metadatas"], limit=1)
        metadata = result.get("metadatas", [])
        return metadata[0] if metadata else None

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        result = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return [
            {"document_id": meta["document_id"], "filename": meta["filename"], "chunk_index": meta["chunk_index"], "page_number": meta.get("page_number"), "text": text, "score": round(1.0 - float(distance), 4)}
            for text, meta, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0])
        ]

    def list_documents(self) -> list[dict]:
        counts: dict[str, dict] = {}
        for meta in self._collection.get(include=["metadatas"]).get("metadatas", []):
            doc_id = meta["document_id"]
            counts.setdefault(doc_id, {"document_id": doc_id, "filename": meta["filename"], "chunks": 0})["chunks"] += 1
        return list(counts.values())

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
