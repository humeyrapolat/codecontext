# CodeContext - AI Codebase Assistant

CodeContext is a FastAPI-based AI assistant that indexes public GitHub repositories and answers natural-language questions about the codebase. It combines repository cloning, language-aware chunking, local embeddings, FAISS retrieval, LangChain tool-calling, Groq-hosted LLMs, LangFuse tracing, and a small RAGAS evaluation entrypoint.

The project started as a single-repository prototype and is being refactored toward a cleaner repo-scoped architecture suitable for portfolio review and production-oriented discussion.

## What It Does

- Indexes public GitHub repositories.
- Stores each indexed repository under a stable `repo_id`.
- Builds a FAISS vector index from supported code and Markdown files.
- Creates an agent runtime per repository.
- Lets users ask questions against a specific indexed repository.
- Keeps conversation memory scoped by repository and session.
- Gives the agent tools for semantic code search, file listing, and file content lookup.
- Keeps generated runtime artifacts out of Git.
- Includes lightweight unit tests and GitHub Actions CI.

## Architecture

```text
GitHub URL
  -> URL validation and repo_id generation
  -> repo-scoped storage paths
  -> Git clone
  -> file filtering
  -> language-aware chunking
  -> local embeddings
  -> FAISS index
  -> AgentRuntime per repository
  -> FastAPI question endpoints
  -> LangChain tool calls
  -> Groq LLM answer
```

Runtime storage is scoped by repository:

```text
data/
  repositories/
    {repo_id}/
      source/
      faiss/
```

## Tech Stack

| Layer                 | Technology                     |
| --------------------- | ------------------------------ |
| API                   | FastAPI, Uvicorn               |
| Agent                 | LangChain tool-calling         |
| LLM                   | Groq, Llama 3.3 70B            |
| Embeddings            | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store          | FAISS                          |
| Repository Access     | GitPython                      |
| Observability         | LangFuse callbacks             |
| Evaluation            | RAGAS entrypoint               |
| Dependency Management | uv                             |
| Tests                 | Python `unittest`              |
| CI                    | GitHub Actions                 |

## Key Design Decisions

**Repo-scoped state**
The first prototype kept one active global agent. CodeContext now stores runtime state by `repo_id`, which prepares the system for multi-repository support and prevents the API layer from owning application state directly.

**AgentRuntime abstraction**
Each agent owns its LLM-with-tools, tools, vectorstore, and repository path. This removes hidden global vectorstore state and makes it clear which index an agent is answering from.

**Lazy model initialization**
Embedding and text-splitting dependencies are loaded when indexing is needed, not during API import. This keeps startup lighter and avoids importing heavy ML dependencies before the application needs them.

**Path safety for file tools**
The file-content tool resolves requested paths against the repository root and rejects paths that escape the indexed repository.

**Honest evaluation**
RAGAS support exists, but this README does not publish benchmark scores until they are generated from a reproducible evaluation run.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Groq API key
- Optional LangFuse credentials for tracing

### Install

```bash
git clone https://github.com/humeyrapolat/codecontext.git
cd codecontext

uv sync
cp .env.example .env
```

Fill in `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key_here
LANGFUSE_SECRET_KEY=your_langfuse_secret_key_here
LANGFUSE_HOST=https://cloud.langfuse.com
```

### Run

```bash
uv run uvicorn app.api:app --reload
```

API docs:

```text
http://localhost:8000/docs
```

## API Examples

### Index a Repository

```http
POST /repositories
```

```json
{
  "repo_url": "https://github.com/humeyrapolat/codecontext"
}
```

Response:

```json
{
  "message": "Repository indexed successfully.",
  "repo_url": "https://github.com/humeyrapolat/codecontext.git",
  "repo_id": "humeyrapolat-codecontext-...",
  "documents_count": 8,
  "chunks_count": 24
}
```

`POST /index` is kept as a backwards-compatible alias.

### Ask a Repository-Scoped Question

```http
POST /repositories/{repo_id}/ask
```

```json
{
  "question": "How does repository indexing work?",
  "session_id": "demo-session"
}
```

### Ask the Active Repository

```http
POST /ask
```

```json
{
  "repo_id": "optional-repo-id",
  "question": "Which files define the agent tools?",
  "session_id": "demo-session"
}
```

If `repo_id` is omitted, CodeContext uses the most recently indexed repository.

### List Indexed Repositories

```http
GET /repositories
```

### Check Status

```http
GET /status
```

### Evaluate

```http
POST /evaluate?repo_id={repo_id}
```

Evaluation output is written to `evaluation_results.json`, which is ignored by Git.

### Clear Conversation Memory

```http
POST /clear
```

```json
{
  "repo_id": "optional-repo-id",
  "session_id": "demo-session"
}
```

## Project Structure

```text
codecontext/
  app/
    api.py          # FastAPI routes and request/response models
    agent.py        # AgentRuntime, tool factory, agent loop
    indexer.py      # clone -> load -> chunk -> embed -> FAISS
    repository.py   # GitHub URL validation, repo_id, storage paths
    state.py        # in-memory repo runtime registry
    evaluation.py   # RAGAS evaluation entrypoint
  tests/
    test_repository.py
    test_state.py
  .github/workflows/ci.yml
  .env.example
  pyproject.toml
  uv.lock
```

## Run Checks

```bash
python3 -m unittest discover -s tests
python3 -m compileall app tests main.py
```

With dependencies installed:

```bash
uv run python -m unittest discover -s tests
uv run python -m compileall app tests main.py
```

## Current Limitations

- Runtime state is still in memory, so it is lost on process restart.
- Indexing still happens inside the request lifecycle; a production system should move it to a background job.
- Conversation memory is repo-scoped but not persisted.
- Private GitHub repositories are not supported yet.
- Retrieval quality needs a larger reproducible evaluation dataset before publishing metrics.

## Next Improvements

- Add background indexing jobs and job status endpoints.
- Add persistent state with Redis or Postgres.
- Add citation metadata with file paths and line ranges.
- Add hybrid retrieval and reranking.
- Add private repo support through GitHub token authentication.
- Add streaming answers.
