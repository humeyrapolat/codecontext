from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.indexer import build_index, load_index
from app.agent import build_agent, ask_agent
from app.evaluation import run_evaluation
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="CodeContext API",
    description="AI-powered codebase assistant",
    version="1.0.0"
)

agent_tuple = None


class IndexRequest(BaseModel):
    repo_url: str


class IndexResponse(BaseModel):
    message: str
    repo_url: str


class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"  # opsiyonel, default "default"


class QuestionResponse(BaseModel):
    question: str
    answer: str
    session_id: str


class ClearSessionRequest(BaseModel):
    session_id: str = "default"

@app.get("/")
async def root():
    return {"status": "CodeContext API çalışıyor 🚀"}


@app.post("/index", response_model=IndexResponse)
async def index_repository(request: IndexRequest):
    global agent_tuple
    
    if not request.repo_url.startswith("https://github.com"):
        raise HTTPException(
            status_code=400,
            detail="Sadece GitHub URL'leri kabul edilir"
        )
    
    print(f"🔄 Repo indexleniyor: {request.repo_url}")
    vectorstore = build_index(request.repo_url)
    agent_tuple = build_agent(vectorstore)
    print("✅ Agent hazır!")
    
    return IndexResponse(
        message="Repo başarıyla indexlendi!",
        repo_url=request.repo_url
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    global agent_tuple
    
    if agent_tuple is None:
        raise HTTPException(
            status_code=400,
            detail="Önce /index endpoint'i ile bir repo indexleyin"
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Soru boş olamaz"
        )
    
    result = ask_agent(agent_tuple, request.question, request.session_id)
    
    return QuestionResponse(
        question=result["question"],
        answer=result["answer"],
        session_id=result["session_id"]
    )

@app.post("/clear")
async def clear_session(request: ClearSessionRequest):
    """Konuşma geçmişini temizle"""
    from app.agent import clear_session
    clear_session(request.session_id)
    return {"message": f"Session '{request.session_id}' temizlendi"}
class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_correctness: float
    context_precision: float
    timestamp: str
    num_questions: int


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_system():
    global agent_tuple
    
    if agent_tuple is None:
        raise HTTPException(
            status_code=400,
            detail="Önce /index endpoint'i ile bir repo indexleyin"
        )
    
    print("🔄 RAGAS evaluation başlatılıyor...")
    
    # RAGAS sync kod — threadpool'da çalıştır
    scores = await run_in_threadpool(run_evaluation, agent_tuple)
    
    return EvaluationResponse(**scores)

@app.get("/status")
async def status():
    return {
        "agent_ready": agent_tuple is not None,
        "index_exists": os.path.exists("faiss_index")
    }