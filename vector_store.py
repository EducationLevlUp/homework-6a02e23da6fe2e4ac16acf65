"""
Модуль векторного хранилища на базе Qdrant.

Инициализирует локальный клиент Qdrant, создаёт коллекцию
и предоставляет экземпляр QdrantVectorStore с эмбеддингами Ollama.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

if TYPE_CHECKING:
    from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
QDRANT_DATA_DIR: str = os.environ.get("QDRANT_DATA_DIR", "qdrant_data")
COLLECTION_NAME: str = os.environ.get("QDRANT_COLLECTION", "rag_collection")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
# nomic-embed-text генерирует 768-мерные векторы
EMBEDDING_DIM: int = 768

# ---------------------------------------------------------------------------
# Глобальные однопоточные экземпляры (ленивая инициализация)
# ---------------------------------------------------------------------------
_client: QdrantClient | None = None
_embeddings: OllamaEmbeddings | None = None
_vector_store: QdrantVectorStore | None = None


def get_client() -> QdrantClient:
    """Возвращает (и при необходимости создаёт) локальный клиент Qdrant."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_DATA_DIR)
    return _client


def get_embeddings() -> OllamaEmbeddings:
    """Возвращает (и при необходимости создаёт) экземпляр OllamaEmbeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def ensure_collection() -> None:
    """Создаёт коллекцию в Qdrant, если она ещё не существует."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def get_vector_store() -> QdrantVectorStore:
    """Возвращает (и при необходимости создаёт) QdrantVectorStore."""
    global _vector_store
    if _vector_store is None:
        ensure_collection()
        _vector_store = QdrantVectorStore(
            client=get_client(),
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )
    return _vector_store


def add_documents(documents: list[Document]) -> list[str]:
    """Добавляет документы в векторное хранилище и возвращает ID."""
    store = get_vector_store()
    return store.add_documents(documents)


def search(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """
    Семантический поиск по базе знаний.

    Возвращает список кортежей (Document, score), где score —
    метрика релевантности (cosine distance, чем меньше — тем лучше).
    """
    store = get_vector_store()
    return store.similarity_search_with_score(query, k=k)


def count_documents() -> int:
    """Возвращает количество документов в коллекции."""
    client = get_client()
    info = client.get_collection(COLLECTION_NAME)
    return info.points_count  # type: ignore[attr-defined]


def reset() -> None:
    """Полностью сбрасывает хранилище (удаляет коллекцию и пересоздаёт)."""
    global _client, _embeddings, _vector_store
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)
    _client = None
    _embeddings = None
    _vector_store = None
    ensure_collection()
