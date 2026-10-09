from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langfuse.langchain import CallbackHandler

from app.repository import resolve_repo_file

load_dotenv()


@dataclass
class AgentRuntime:
    """Runtime dependencies owned by one codebase assistant agent."""

    llm_with_tools: Any
    tools: list[Any]
    vectorstore: FAISS
    repo_path: Path
    repo_id: str


# Session memory is scoped by both repository and session id.
# This keeps conversations for different repositories from leaking into each other.
_conversation_history: dict[str, list[Any]] = {}


def build_session_key(repo_id: str, session_id: str) -> str:
    return f"{repo_id}:{session_id}"


def get_or_create_session(repo_id: str, session_id: str) -> list[Any]:
    """Return the message history for one repo/session pair."""
    key = build_session_key(repo_id, session_id)
    if key not in _conversation_history:
        _conversation_history[key] = [
            SystemMessage(
                content=(
                    "You are an expert codebase assistant. "
                    "You help developers understand codebases by searching through code. "
                    "You must always use the provided tools to search before answering. "
                    "Remember the conversation history and refer to previous questions when relevant. "
                    "Answer in the same language as the question."
                )
            )
        ]
    return _conversation_history[key]


def clear_session(session_id: str, repo_id: str = "default") -> None:
    """Clear the message history for one repo/session pair."""
    key = build_session_key(repo_id, session_id)
    _conversation_history.pop(key, None)


def build_tools(vectorstore: FAISS, repo_path: Path) -> list[Any]:
    """Create tools bound to a specific repository and vectorstore."""

    @tool
    def search_code(query: str) -> str:
        """
        Search the codebase for relevant code snippets.
        Use this when the user asks about functions, classes, implementations, or code logic.
        """
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No relevant code found."

        results = []
        for doc in docs:
            source = doc.metadata.get("relative_path", "unknown")
            results.append(f"--- File: {source} ---\n{doc.page_content}")

        return "\n\n".join(results)

    @tool
    def list_files(directory: str = "") -> str:
        """
        List indexed source and Markdown files in the repository.
        Use this when the user asks about project structure or available files.
        """
        if not repo_path.exists():
            return "Error: No repository indexed yet."

        supported_suffixes = {".py", ".js", ".ts", ".java", ".kt", ".md"}
        ignored_parts = {"node_modules", "__pycache__", ".venv"}
        files = []

        for file in repo_path.rglob("*"):
            if file.suffix not in supported_suffixes:
                continue
            if any(part.startswith(".") or part in ignored_parts for part in file.parts):
                continue

            relative_path = str(file.relative_to(repo_path))
            if directory and directory not in relative_path:
                continue

            files.append(relative_path)

        if not files:
            return "No files found."

        return f"Repository contains {len(files)} files:\n" + "\n".join(sorted(files))

    @tool
    def get_file_content(file_path: str) -> str:
        """
        Get the content of a specific file.
        Input should be a relative path such as 'src/auth.py'.
        """
        try:
            full_path = resolve_repo_file(repo_path, file_path)
        except ValueError as exc:
            return f"Error: {exc}"

        if not full_path.exists():
            return f"Error: File '{file_path}' not found."

        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return f"Error reading file: {exc}"

        if len(content) > 3000:
            content = content[:3000] + "\n\n... [File truncated]"

        return f"Content of {file_path}:\n\n{content}"

    return [search_code, list_files, get_file_content]


def build_agent(
    vectorstore: FAISS,
    repo_path: str = "indexed_repo",
    repo_id: str = "default",
) -> AgentRuntime:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
    tools = build_tools(vectorstore, Path(repo_path))
    llm_with_tools = llm.bind_tools(tools)

    return AgentRuntime(
        llm_with_tools=llm_with_tools,
        tools=tools,
        vectorstore=vectorstore,
        repo_path=Path(repo_path),
        repo_id=repo_id,
    )


def ask_agent(agent_runtime: AgentRuntime, question: str, session_id: str = "default") -> dict[str, str]:
    """Ask a repo-scoped agent a question and preserve session memory."""
    llm_with_tools = agent_runtime.llm_with_tools
    tool_map = {tool_instance.name: tool_instance for tool_instance in agent_runtime.tools}
    langfuse_handler = CallbackHandler()

    messages = get_or_create_session(agent_runtime.repo_id, session_id)
    messages.append(HumanMessage(content=question))

    try:
        response = llm_with_tools.invoke(
            messages,
            config={"callbacks": [langfuse_handler]},
        )
        messages.append(response)

        max_iterations = 5
        iteration = 0

        while getattr(response, "tool_calls", None) and iteration < max_iterations:
            iteration += 1

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_result = (
                    tool_map[tool_name].invoke(tool_args)
                    if tool_name in tool_map
                    else f"Tool '{tool_name}' not found"
                )

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

            response = llm_with_tools.invoke(
                messages,
                config={"callbacks": [langfuse_handler]},
            )
            messages.append(response)

        answer = response.content

    except Exception as exc:
        print(f"Tool-calling failed; using retrieval fallback: {exc}")

        plain_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        context = tool_map["search_code"].invoke({"query": question})
        fallback_messages = [
            SystemMessage(
                content="You are a codebase assistant. Answer based on the provided code context."
            ),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]
        fallback_response = plain_llm.invoke(fallback_messages)
        answer = fallback_response.content
        messages.append(AIMessage(content=answer))

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "repo_id": agent_runtime.repo_id,
    }