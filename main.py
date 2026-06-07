# ruff: noqa: I001
from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from tools import TOOLS

# ---------------------------------------------------------------------------
# Интерактивный CLI-клиент для агента с RAG-памятью.
#
# Команды:
#     /add <текст> <заголовок>   — добавить документ в базу знаний
#     /search <запрос>           — поиск по базе знаний
#     /quit                      — выход
#
# Любой другой ввод обрабатывается агентом, который может
# самостоятельно решить, использовать ли RAG-инструменты.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
LLM_MODEL: str = "llama3"
SYSTEM_PROMPT: str = (
    "Ты — полезный ассистент с доступом к базе знаний.\n"
    "Используй инструмент search_knowledge_base для поиска информации.\n"
    "Используй инструмент add_to_knowledge_base для сохранения новых знаний.\n"
    "Отвечай на русском языке, если пользователь пишет на русском.\n"
    "Если в базе знаний нет релевантной информации, честно скажи об этом."
)

# ---------------------------------------------------------------------------
# Инициализация агента
# ---------------------------------------------------------------------------


def create_agent():
    """
    Создаёт агента через create_react_agent (современный аналог create_agent
    в LangChain v0.2+).

    Агент получает системный промпт и набор RAG-инструментов.
    """
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )
    return agent


# ---------------------------------------------------------------------------
# CLI-цикл
# ---------------------------------------------------------------------------


def print_welcome() -> None:
    print("=" * 60)
    print("  🤖 Агент с RAG-памятью (Qdrant + Ollama)")
    print("=" * 60)
    print()
    print("Команды:")
    print("  /add <текст> | <заголовок>  — добавить документ")
    print("  /search <запрос>          — поиск по базе знаний")
    print("  /quit                     — выход")
    print()
    print("Любой другой текст будет обработан агентом.")
    print("-" * 60)


def parse_command(line: str) -> tuple[str, str]:
    """
    Парсит команду и аргументы.

    Возвращает (command, args).
    """
    parts = line.strip().split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    return command, args


def handle_add(args: str, agent) -> None:
    """Обработка команды /add."""
    # Формат: /add "текст" | "заголовок"  или  /add текст | заголовок
    if "|" in args:
        content, title = args.split("|", 1)
        content = content.strip().strip('"').strip("'")
        title = title.strip().strip('"').strip("'")
    else:
        # Если нет разделителя, просим пользователя
        content = args.strip().strip('"').strip("'")
        if not content:
            print("❌ Укажите текст документа.")
            return
        title = input("  Введите заголовок: ").strip()
        if not title:
            title = "Без названия"

    # Используем инструмент напрямую
    result = TOOLS[1].invoke({"content": content, "title": title})
    print(f"  {result}")


def handle_search(args: str, agent) -> None:
    """Обработка команды /search."""
    query = args.strip().strip('"').strip("'")
    if not query:
        print("❌ Укажите поисковый запрос.")
        return

    # Используем инструмент напрямую
    result = TOOLS[0].invoke({"query": query, "max_results": 4})
    print(f"  {result}")


def handle_agent_query(text: str, agent) -> None:
    """Передаёт запрос агенту."""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=text),
    ]
    try:
        response = agent.invoke({"messages": messages})
        # Извлекаем последний AI-ответ
        ai_msg = response["messages"][-1]
        print(f"  {ai_msg.content}")
    except Exception as exc:
        print(f"  ❌ Ошибка агента: {exc}")


def main() -> None:
    print_welcome()

    try:
        agent = create_agent()
        print("✅ Агент инициализирован.\n")
    except Exception as exc:
        print(f"❌ Не удалось инициализировать агента: {exc}")
        print("   Убедитесь, что Ollama запущен и модели загружены.")
        sys.exit(1)

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 До свидания!")
            break

        if not line:
            continue

        command, args = parse_command(line)

        if command == "/quit":
            print("👋 До свидания!")
            break
        elif command == "/add":
            handle_add(args, agent)
        elif command == "/search":
            handle_search(args, agent)
        else:
            handle_agent_query(line, agent)


if __name__ == "__main__":
    main()
