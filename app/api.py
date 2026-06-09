from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from dotenv import load_dotenv

from app.agent import ask_agent, build_agent, clear_session as clear_agent_session
from app.indexer import build_index
from app.repository import build_repository_paths, parse_github_repo_url
from app.state import RepositoryRuntime, app_state

load_dotenv()

app = FastAPI(
    title="CodeContext API",
    description="AI-powered codebase assistant",
    version="1.0.0",
)


class IndexRequest(BaseModel):
    repo_url: str


class IndexResponse(BaseModel):
    message: str
    repo_url: str
    repo_id: str
    documents_count: int
    chunks_count: int


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"
    repo_id: str | None = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    session_id: str
    repo_id: str


class ClearSessionRequest(BaseModel):
    session_id: str = "default"
    repo_id: str | None = None


class RepositorySummary(BaseModel):
    repo_id: str
    repo_url: str


class RepositoriesResponse(BaseModel):
    repositories: list[RepositorySummary]
    active_repo_id: str | None


class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_correctness: float
    context_precision: float
    timestamp: str
    num_questions: int


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "CodeContext API is running"}


def require_agent_or_404(repo_id: str | None = None) -> RepositoryRuntime:
    try:
        return app_state.require_repository(repo_id)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=404,
            detail="Repository is not indexed yet",
        ) from exc


async def index_repository_request(request: IndexRequest) -> IndexResponse:
    try:
        repo_ref = parse_github_repo_url(request.repo_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    repo_paths = build_repository_paths(repo_ref.repo_id)

    index_result = await run_in_threadpool(
        build_index,
        repo_ref.normalized_url,
        repo_paths.source_dir,
        repo_paths.index_dir,
    )

    agent_runtime = build_agent(
        index_result.vectorstore,
        str(index_result.repo_path),
        repo_id=repo_ref.repo_id,
    )

    app_state.set_repository(
        repo_ref.repo_id,
        repo_ref.normalized_url,
        agent_runtime,
    )

    return IndexResponse(
        message="Repository indexed successfully.",
        repo_url=repo_ref.normalized_url,
        repo_id=repo_ref.repo_id,
        documents_count=index_result.documents_count,
        chunks_count=index_result.chunks_count,
    )


@app.post("/repositories", response_model=IndexResponse)
async def create_repository(request: IndexRequest) -> IndexResponse:
    return await index_repository_request(request)


@app.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest) -> IndexResponse:
    return await index_repository_request(request)


@app.get("/repositories", response_model=RepositoriesResponse)
async def list_repositories() -> RepositoriesResponse:
    return RepositoriesResponse(
        repositories=[
            RepositorySummary(repo_id=repo.repo_id, repo_url=repo.repo_url)
            for repo in app_state.list_repositories()
        ],
        active_repo_id=app_state.active_repo_id,
    )


def answer_question(question: str, session_id: str, repo_id: str | None = None) -> QuestionResponse:
    repository = require_agent_or_404(repo_id)

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    result = ask_agent(
        repository.agent_runtime,
        question,
        session_id,
    )

    return QuestionResponse(
        question=result["question"],
        answer=result["answer"],
        session_id=result["session_id"],
        repo_id=repository.repo_id,
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest) -> QuestionResponse:
    return answer_question(
        question=request.question,
        session_id=request.session_id,
        repo_id=request.repo_id,
    )


@app.post("/repositories/{repo_id}/ask", response_model=QuestionResponse)
async def ask_repository(repo_id: str, request: QuestionRequest) -> QuestionResponse:
    return answer_question(
        question=request.question,
        session_id=request.session_id,
        repo_id=repo_id,
    )


@app.post("/clear")
async def clear_session(request: ClearSessionRequest) -> dict[str, str]:
    repository = require_agent_or_404(request.repo_id)
    clear_agent_session(request.session_id, repo_id=repository.repo_id)

    return {"message": f"Session '{request.session_id}' cleared"}


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system(repo_id: str | None = None) -> EvaluationResponse:
    repository = require_agent_or_404(repo_id)

    from app.evaluation import run_evaluation

    scores = await run_in_threadpool(run_evaluation, repository.agent_runtime)

    return EvaluationResponse(**scores)


@app.get("/status")
async def status() -> dict[str, str | int | bool | None]:
    return {
        "agent_ready": app_state.agent_ready,
        "active_repo_id": app_state.active_repo_id,
        "indexed_repositories": len(app_state.repositories),
    }