from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

_vectorstore: FAISS | None = None

# ============================================================
# CONVERSATION MEMORY
# ============================================================
# Her session için mesaj geçmişini tutuyoruz.
# session_id → messages listesi
# Neden dict? İleride birden fazla kullanıcı olabilir,
# her biri kendi geçmişine sahip olmalı.
# ============================================================
_conversation_history: dict[str, list] = {}


def set_vectorstore(vs: FAISS):
    global _vectorstore
    _vectorstore = vs


def get_or_create_session(session_id: str) -> list:
    """Session yoksa oluştur, varsa getir"""
    if session_id not in _conversation_history:
        _conversation_history[session_id] = [
            SystemMessage(content="""You are an expert codebase assistant. 
You help developers understand codebases by searching through code.
You must always use the provided tools to search before answering.
Remember the conversation history and refer to previous questions when relevant.
Answer in the same language as the question.""")
        ]
    return _conversation_history[session_id]


def clear_session(session_id: str):
    """Session geçmişini temizle"""
    if session_id in _conversation_history:
        del _conversation_history[session_id]


@tool
def search_code(query: str) -> str:
    """
    Search the codebase for relevant code snippets.
    Use this when the user asks about specific functions,
    classes, implementations, or code logic.
    """
    if _vectorstore is None:
        return "Error: No codebase indexed yet."
    docs = _vectorstore.similarity_search(query, k=5)
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
    List all indexed files in the repository.
    Use this when the user asks about project structure or what files exist.
    """
    if not os.path.exists("indexed_repo"):
        return "Error: No repository indexed yet."
    repo_path = Path("indexed_repo")
    supported = {".py", ".js", ".ts", ".java", ".kt", ".md"}
    files = []
    for f in repo_path.rglob("*"):
        if f.suffix not in supported:
            continue
        if any(part.startswith(".") or part in ["node_modules", "__pycache__", ".venv"]
               for part in f.parts):
            continue
        relative = str(f.relative_to(repo_path))
        if directory and directory not in relative:
            continue
        files.append(relative)
    if not files:
        return "No files found."
    return f"Repository contains {len(files)} files:\n" + "\n".join(sorted(files))


@tool
def get_file_content(file_path: str) -> str:
    """
    Get the full content of a specific file.
    Use this when the user asks to see a specific file.
    Input should be the relative path (e.g. 'src/auth.py').
    """
    full_path = Path("indexed_repo") / file_path
    if not full_path.exists():
        return f"Error: File '{file_path}' not found."
    try:
        content = full_path.read_text(encoding="utf-8", errors="ignore")
        if len(content) > 3000:
            content = content[:3000] + "\n\n... [File truncated]"
        return f"Content of {file_path}:\n\n{content}"
    except Exception as e:
        return f"Error reading file: {e}"


def build_agent(vectorstore: FAISS):
    set_vectorstore(vectorstore)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )
    tools = [search_code, list_files, get_file_content]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools


def ask_agent(agent_tuple, question: str, session_id: str = "default") -> dict:
    """
    Agent'a soru sor.
    session_id ile konuşma geçmişi takip edilir.
    Aynı session_id → önceki sorular hatırlanır.
    """
    llm_with_tools, tools = agent_tuple
    langfuse_handler = CallbackHandler()
    tool_map = {t.name: t for t in tools}

    # Session geçmişini al veya oluştur
    messages = get_or_create_session(session_id)
    
    # Kullanıcının sorusunu geçmişe ekle
    messages.append(HumanMessage(content=question))

    try:
        response = llm_with_tools.invoke(
            messages,
            config={"callbacks": [langfuse_handler]}
        )
        messages.append(response)

        max_iterations = 5
        iteration = 0

        while hasattr(response, 'tool_calls') and response.tool_calls and iteration < max_iterations:
            iteration += 1
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                if tool_name in tool_map:
                    tool_result = tool_map[tool_name].invoke(tool_args)
                else:
                    tool_result = f"Tool '{tool_name}' not found"
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
            response = llm_with_tools.invoke(
                messages,
                config={"callbacks": [langfuse_handler]}
            )
            messages.append(response)

        answer = response.content

    except Exception as e:
        print(f"⚠️ Tool call hatası, fallback: {e}")
        plain_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        context = search_code.invoke({"query": question})
        fallback_messages = [
            SystemMessage(content="You are a codebase assistant. Answer based on the provided code context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
        ]
        fallback_response = plain_llm.invoke(fallback_messages)
        answer = fallback_response.content
        # Fallback cevabı da geçmişe ekle
        messages.append(AIMessage(content=answer))

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id
    }