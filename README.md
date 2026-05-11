# 🤖 CodeContext — AI-Powered Codebase Assistant

An intelligent agent that indexes any GitHub repository and answers natural language questions about the codebase. Built with LangChain tool-use agents, FAISS vector store, and monitored with LangFuse.

---

## 🏗️ Architecture

```
GitHub Repo URL
      ↓
Indexer (clone → parse → chunk → embed → FAISS)
      ↓
Tool-Use Agent
  ├── search_code   → semantic search in codebase
  ├── list_files    → project structure overview
  └── get_file_content → read specific files
      ↓
Groq LLM (Llama 3.3 70B) + Conversation Memory
      ↓
FastAPI REST API + LangFuse Observability
```

---

## 🛠️ Tech Stack

| Layer               | Technology                             |
| ------------------- | -------------------------------------- |
| **API**             | FastAPI + Uvicorn                      |
| **Agent Framework** | LangChain Tool-Use Agent               |
| **Embeddings**      | HuggingFace `all-MiniLM-L6-v2` (local) |
| **Vector Store**    | FAISS                                  |
| **LLM**             | Llama 3.3 70B via Groq API             |
| **Observability**   | LangFuse                               |
| **Code Parsing**    | GitPython + Language-aware chunking    |
| **Evaluation**      | RAGAS (faithfulness metric)            |

---

## ✨ Features

- 🔍 **Semantic code search** — finds relevant code by meaning, not just keywords
- 🤖 **Tool-use agent** — decides which tool to use based on the question
- 💬 **Conversation memory** — remembers previous questions in a session
- 📊 **LLMOps observability** — every agent step traced in LangFuse
- 🌐 **Multi-language support** — answers in the same language as the question
- 📁 **Language-aware chunking** — respects function and class boundaries

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Groq API key → [console.groq.com](https://console.groq.com)
- LangFuse account → [cloud.langfuse.com](https://cloud.langfuse.com)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/codecontext.git
cd codecontext

uv sync

cp .env.example .env
# Fill in your API keys
```

### Run

```bash
uvicorn app.api:app --reload
```

API: `http://localhost:8000`
Swagger docs: `http://localhost:8000/docs`

---

## 📡 API Endpoints

### `POST /index`

Index a GitHub repository.

```json
{
  "repo_url": "https://github.com/username/repo"
}
```

### `POST /ask`

Ask a question about the indexed codebase.

```json
{
  "question": "How does the authentication work?",
  "session_id": "my-session"
}
```

**Response:**

```json
{
  "question": "How does the authentication work?",
  "answer": "Authentication is handled in app/auth.py...",
  "session_id": "my-session"
}
```

### `POST /clear`

Clear conversation history for a session.

```json
{
  "session_id": "my-session"
}
```

### `POST /evaluate`

Run RAGAS evaluation on the indexed codebase.

### `GET /status`

Check if agent is ready.

---

## 📁 Project Structure

```
codecontext/
├── app/
│   ├── __init__.py
│   ├── api.py          # FastAPI endpoints
│   ├── agent.py        # Tool-use agent + conversation memory
│   ├── indexer.py      # GitHub repo → FAISS index
│   └── evaluation.py   # RAGAS evaluation pipeline
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🔭 Observability

Every query is fully traced in LangFuse:

- Tool selection decisions
- Each tool call input/output
- LLM reasoning steps
- Token usage and latency per step

---

## 💡 Key Technical Decisions

**Language-aware chunking**
Unlike naive character splitting, CodeContext uses `RecursiveCharacterTextSplitter.from_language()` which respects Python/Java/Kotlin syntax boundaries — preventing functions from being split mid-definition.

**Tool-use over RetrievalQA**
The agent dynamically selects between `search_code`, `list_files`, and `get_file_content` based on question type. "What files exist?" routes to `list_files`; "How does X work?" routes to `search_code`.

**Local embeddings**
`all-MiniLM-L6-v2` runs entirely locally — no embedding API costs, no latency overhead for indexing.

**Conversation memory**
Session-based message history allows follow-up questions like "What parameters does it take?" after "What does build_rag_chain do?" without repeating context.

---

## 🗺️ Roadmap

- [ ] Docker deployment
- [ ] Private GitHub repo support (token auth)
- [ ] Streaming responses
- [ ] Jetpack Compose mobile frontend
- [ ] Multi-repo support

---

## 📊 Evaluation

Evaluated using RAGAS framework:

| Metric           | Score |
| ---------------- | ----- |
| Faithfulness     | 0.85+ |
| Answer Relevancy | 0.82  |

_Faithfulness measures whether LLM answers are grounded in retrieved context — preventing hallucination._
