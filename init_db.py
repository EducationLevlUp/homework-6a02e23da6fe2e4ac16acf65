#!/usr/bin/env python3
"""
Скрипт инициализации: загружает текстовые документы из указанной директории
в векторное хранилище ChromaDB.

Использование:
    python init_db.py <путь_к_директории>
    python init_db.py                          # по умолчанию: sample_docs/
"""

from __future__ import annotations

import sys
from pathlib import Path

from chunking import split_file
from vector_store import add_documents, count_documents, ensure_collection

DEFAULT_DOCS_DIR: str = "sample_docs"


def load_directory(docs_dir: str) -> None:
    """
    Сканирует директорию на наличие .txt файлов, разбивает каждый на чанки
    и добавляет в векторное хранилище.
    """
    path = Path(docs_dir)
    if not path.is_dir():
        print(f"❌ Директория не найдена: {docs_dir}")
        sys.exit(1)

    txt_files = sorted(path.glob("*.txt"))
    if not txt_files:
        print(f"⚠️  В директории {docs_dir} нет .txt файлов.")
        return

    ensure_collection()
    total_chunks = 0

    for file_path in txt_files:
        try:
            chunks = split_file(str(file_path))
            add_documents(chunks)
            total_chunks += len(chunks)
            print(f"  ✅ {file_path.name} → {len(chunks)} чанков")
        except Exception as exc:
            print(f"  ❌ Ошибка при обработке {file_path.name}: {exc}")

    print(f"\n📊 Загружено {total_chunks} чанков из {len(txt_files)} файлов.")
    print(f"📦 Всего документов в коллекции: {count_documents()}")


def main() -> None:
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCS_DIR
    print(f"🚀 Инициализация базы знаний из директории: {docs_dir}")
    load_directory(docs_dir)


if __name__ == "__main__":
    main()
