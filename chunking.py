"""
Система чанкинга документов.

Использует RecursiveCharacterTextSplitter для разбиения текстов
на чанки с сохранением метаданных (title, source и т.д.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Константы чанкинга
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200


def get_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Возвращает настроенный экземпляр RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_text(
    text: str,
    title: str = "Untitled",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """
    Разбивает текст на чанки-документы с метаданными.

    Параметры
    ----------
    text : str
        Исходный текст для разбиения.
    title : str
        Заголовок документа (сохраняется в metadata каждого чанка).
    chunk_size : int
        Максимальный размер чанка в символах.
    chunk_overlap : int
        Перекрытие между соседними чанками.
    extra_metadata : dict | None
        Дополнительные метаданные для каждого чанка.

    Возвращает
    ----------
    list[Document]
        Список чанков-документов.
    """
    splitter = get_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    base_metadata: dict[str, Any] = {"title": title}
    if extra_metadata:
        base_metadata.update(extra_metadata)

    documents = splitter.create_documents(
        texts=[text],
        metadatas=[base_metadata],
    )
    return documents


def split_file(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Читает текстовый файл и разбивает его содержимое на чанки.

    Имя файла используется как title в метаданных.
    """
    with open(file_path, encoding="utf-8") as fh:
        text = fh.read()

    title = Path(file_path).stem
    return split_text(
        text=text,
        title=title,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata={"source": file_path},
    )
