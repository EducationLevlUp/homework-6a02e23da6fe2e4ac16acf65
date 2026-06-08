"""
Модуль векторного хранилища на базе ChromaDB.

Инициализирует локальное хранилище ChromaDB, создаёт коллекцию
и предоставляет экземпляр ChromaVectorStore с эмбеддингами Ollama.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

if TYPE_CHECKING:
    from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
CHROMA_DATA_DIR: str = os.environ.get("CHROMA_DATA_DIR", "chroma_data")
COLLECTION_NAME: str = os.environ.get("CHROMA_COLLECTION", "rag_collection")
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Глобальные однопоточные экземпляры (ленивая инициализация)
# ---------------------------------------------------------------------------
_embeddings: OllamaEmbeddings | None = None
_vector_store: Chroma | None = None


def get_embeddings() -> OllamaEmbeddings:
    """Возвращает (и при необходимости создаёт) экземпляр OllamaEmbeddings."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return _embeddings


def ensure_collection() -> None:
    """
    Гарантирует, что коллекция ChromaDB инициализирована.

    ChromaDB создаёт коллекцию автоматически при первом обращении,
    поэтому достаточно вызвать get_vector_store().
    """
    get_vector_store()


def get_vector_store() -> Chroma:
    """Возвращает (и при необходимости создаёт) ChromaVectorStore."""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DATA_DIR,
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
    store = get_vector_store()
    return len(store)


def reset() -> None:
    """Полностью сбрасывает хранилище (удаляет коллекцию и пересоздаёт)."""
    global _vector_store
    store = get_vector_store()
    store.delete_collection()
    _vector_store = None
    # Пересоздаём коллекцию
    ensure_collection()
