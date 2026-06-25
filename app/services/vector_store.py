"""
Vector store: persists chunk embeddings and runs similarity search.

Backed by ChromaDB in persistent mode, so ingested documents survive restarts
without needing a separate database server — ideal for a portfolio MVP.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings


class VectorStore:
    """Stores and queries chunk embeddings in a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        """
        Args:
            persist_dir: Directory where ChromaDB writes its data.
            collection_name: Name of the collection to use/create.
        """
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """
        Stores a document's chunks together with their embeddings.

        Args:
            document_id: Stable id of the source document.
            filename: Original file name (stored as metadata).
            chunks: The chunk texts.
            embeddings: One embedding per chunk, in the same order.
        """
        ids = [f"{document_id}::chunk::{index}" for index in range(len(chunks))]
        metadatas = [
            {"document_id": document_id, "filename": filename, "chunk_index": index}
            for index in range(len(chunks))
        ]
        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """
        Finds the chunks most similar to a query embedding.

        Args:
            query_embedding: The embedded query.
            top_k: How many chunks to return.

        Returns:
            A list of dicts with keys ``document_id``, ``filename``,
            ``chunk_index``, ``text`` and ``score`` (1 - cosine distance).
        """
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: list[dict] = []
        for text, meta, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "document_id": meta["document_id"],
                    "filename": meta["filename"],
                    "chunk_index": meta["chunk_index"],
                    "text": text,
                    # Convert cosine distance to a similarity score in [0, 1].
                    "score": round(1.0 - float(distance), 4),
                }
            )
        return matches

    def list_documents(self) -> list[dict]:
        """
        Summarizes the stored documents and how many chunks each has.

        Returns:
            A list of dicts with ``document_id``, ``filename`` and ``chunks``.
        """
        stored = self._collection.get(include=["metadatas"])
        counts: dict[str, dict] = {}
        for meta in stored.get("metadatas", []):
            doc_id = meta["document_id"]
            if doc_id not in counts:
                counts[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta["filename"],
                    "chunks": 0,
                }
            counts[doc_id]["chunks"] += 1
        return list(counts.values())

    def delete_document(self, document_id: str) -> None:
        """
        Removes every chunk belonging to a document.

        Args:
            document_id: The id of the document to delete.
        """
        self._collection.delete(where={"document_id": document_id})
