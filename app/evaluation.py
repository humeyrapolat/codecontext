import json
import math
import time
from datetime import UTC, datetime
from typing import Any

from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_correctness, context_precision, faithfulness
from ragas.run_config import RunConfig

from app.agent import AgentRuntime

load_dotenv()


TEST_QUESTIONS = [
    {
        "question": "Which vector store is used?",
        "ground_truth": "FAISS vector store is used.",
    },
    {
        "question": "How does the ask endpoint work?",
        "ground_truth": (
            "It receives a question, retrieves relevant code context, "
            "and uses an LLM-backed agent to generate an answer."
        ),
    },
    {
        "question": "What is the API framework used?",
        "ground_truth": "FastAPI is used as the API framework.",
    },
    {
        "question": "What embedding model does CodeContext use?",
        "ground_truth": (
            "A local HuggingFace embedding model, all-MiniLM-L6-v2, "
            "loaded lazily so API startup stays lightweight."
        ),
    },
    {
        "question": "Which LLM provider powers the agent's answers?",
        "ground_truth": "Groq-hosted openai/gpt-oss-120b.",
    },
    {
        "question": "What is repo_id used for?",
        "ground_truth": (
            "A stable per-repository identifier used to scope storage paths "
            "and runtime state, so multiple repositories can be indexed and "
            "queried independently."
        ),
    },
    {
        "question": "How is repo_id generated from a GitHub URL?",
        "ground_truth": (
            "parse_github_repo_url normalizes the URL, builds a slug from the "
            "owner and repo name, and appends an 8-character sha1 hash of the "
            "normalized URL to produce the repo_id."
        ),
    },
    {
        "question": "What happens if a non-GitHub URL is passed to parse_github_repo_url?",
        "ground_truth": (
            "It raises a ValueError, because only https://github.com/{owner}/{repo} "
            "URLs are supported."
        ),
    },
    {
        "question": "What does resolve_repo_file protect against?",
        "ground_truth": (
            "Path traversal: it resolves a requested file path against the "
            "repository root and raises a ValueError if the resolved path "
            "escapes that root."
        ),
    },
    {
        "question": "Which function builds the per-repository storage paths?",
        "ground_truth": (
            "build_repository_paths, which returns a RepositoryPaths with a "
            "source_dir and index_dir under data/repositories/{repo_id}."
        ),
    },
    {
        "question": "What tools does the agent have access to?",
        "ground_truth": (
            "search_code for semantic search over the FAISS index, list_files "
            "to list repository files, and get_file_content to read a specific "
            "file's contents."
        ),
    },
    {
        "question": "Which file defines the agent's tools?",
        "ground_truth": "app/agent.py, in the build_tools function.",
    },
    {
        "question": "What file extensions does the indexer support?",
        "ground_truth": ".py, .js, .ts, .java, .kt, and .md files.",
    },
    {
        "question": "Which directories does the indexer ignore while walking a repository?",
        "ground_truth": (
            "node_modules, __pycache__, and .venv, along with any hidden "
            "(dot-prefixed) path segment."
        ),
    },
    {
        "question": "What splitter does the indexer use for source code files?",
        "ground_truth": (
            "A language-aware RecursiveCharacterTextSplitter built with "
            "from_language for Python, JS, Java, and Kotlin, using a chunk "
            "size of 1000 and chunk overlap of 100; other files use the "
            "generic RecursiveCharacterTextSplitter with the same settings."
        ),
    },
    {
        "question": "How large can a single file be before the indexer skips it?",
        "ground_truth": "Files larger than 100,000 characters (MAX_FILE_SIZE_CHARS) are skipped.",
    },
    {
        "question": "What does app/state.py do?",
        "ground_truth": (
            "It keeps an in-memory registry of RepositoryRuntime objects keyed "
            "by repo_id, tracks which repository is active, and raises when a "
            "requested repository hasn't been indexed yet."
        ),
    },
    {
        "question": "Why was repo-scoped state introduced instead of one global agent?",
        "ground_truth": (
            "To support multiple indexed repositories at once and to stop the "
            "API layer from owning application state directly."
        ),
    },
    {
        "question": "What does the evaluation module measure?",
        "ground_truth": "Faithfulness, answer correctness, and context precision, using RAGAS.",
    },
    {
        "question": "Which LLM is used as the RAGAS judge model?",
        "ground_truth": (
            "The same Groq-hosted openai/gpt-oss-20b model that generates "
            "the evaluation answers."
        ),
    },
    {
        "question": "Where are evaluation results written?",
        "ground_truth": "To evaluation_results.json, which is ignored by Git.",
    },
    {
        "question": "Is CodeContext's FAISS index persisted across process restarts?",
        "ground_truth": (
            "The FAISS index itself is saved to a local index directory, but "
            "the in-memory app_state runtime registry is not, so after a "
            "restart the index must be reloaded and the agent runtime rebuilt."
        ),
    },
    {
        "question": "Can CodeContext index a private GitHub repository?",
        "ground_truth": "No, private repository support is listed as not implemented yet.",
    },
    {
        "question": "Which HTTP method and path index a new repository?",
        "ground_truth": "POST /repositories (POST /index is kept as a backward-compatible alias).",
    },
    {
        "question": "What does GET /status return?",
        "ground_truth": (
            "Whether the agent is ready, the active repo_id, and how many "
            "repositories are currently indexed."
        ),
    },
    {
        "question": "How is conversation memory scoped?",
        "ground_truth": (
            "Per repository and per session, using a session key that "
            "combines repo_id and session_id."
        ),
    },
    {
        "question": "What happens when /ask is called without a repo_id?",
        "ground_truth": "CodeContext falls back to the most recently indexed (active) repository.",
    },
    {
        "question": "What testing framework does the project use?",
        "ground_truth": "Python's built-in unittest, run via unittest discover.",
    },
    {
        "question": "Does the CI pipeline run the test suite?",
        "ground_truth": (
            "Yes, the GitHub Actions workflow installs dependencies, compiles "
            "the app, and runs the unit test suite on every push and pull "
            "request."
        ),
    },
    {
        "question": "What is the main known limitation around evaluation?",
        "ground_truth": (
            "The evaluation set is small, so retrieval quality needs a larger, "
            "reproducible dataset before publishing headline metrics."
        ),
    },
]


def get_contexts_for_question(agent_runtime: AgentRuntime, question: str) -> list[str]:
    docs = agent_runtime.vectorstore.similarity_search(question, k=3)
    return [doc.page_content for doc in docs]


def safe_float(value: Any) -> float:
    if isinstance(value, list):
        selected = float(value[0]) if value else 0.0
    else:
        selected = float(value)

    return round(0.0 if math.isnan(selected) else selected, 3)


def generate_answer_from_context(question: str, context_text: str) -> str:
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=512,
        # Groq's free tier caps this model at a low tokens-per-minute budget.
        # 30 sequential questions plus RAGAS's own ~90 scoring calls burn
        # through that fast; let LangChain's built-in backoff ride out the
        # 429s instead of failing each question outright.
        max_retries=6,
    )
    messages = [
        SystemMessage(
            content=(
                "You are a code assistant. Answer the question using only the provided context. "
                "Give a clear answer and avoid unsupported claims."
            )
        ),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"),
    ]

    response = llm.invoke(messages)
    return response.content


def build_evaluation_dataset(agent_runtime: AgentRuntime) -> Dataset:
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for index, test_case in enumerate(TEST_QUESTIONS):
        question = test_case["question"]
        context_list = get_contexts_for_question(agent_runtime, question)
        context_text = "\n\n".join(context_list)

        if index > 0:
            # Space out requests to stay under Groq's free-tier
            # tokens-per-minute limit instead of bursting through it.
            time.sleep(2)

        try:
            answer = generate_answer_from_context(question, context_text)
        except Exception as exc:
            print(f"Evaluation answer generation failed for question '{question}': {exc}")
            answer = "Could not generate answer."

        questions.append(question)
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(test_case["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation(agent_runtime: AgentRuntime) -> dict[str, float | str | int]:
    """Run a small RAGAS evaluation for smoke-testing retrieval and answer quality.

    This is a lightweight evaluation entrypoint, not a published benchmark.
    Use a larger curated dataset before reporting headline scores in a CV.
    """
    dataset = build_evaluation_dataset(agent_runtime)

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            max_retries=6,
        )
    )
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            context_precision,
            answer_correctness,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        # RAGAS defaults to 16 concurrent LLM calls, which instantly blows
        # through Groq's free-tier tokens-per-minute budget. Run fully
        # sequential instead, with generous retry/backoff — slower, but
        # it actually finishes on a free account.
        run_config=RunConfig(max_workers=1, max_retries=10, max_wait=60),
    )

    scores = {
        "faithfulness": safe_float(results["faithfulness"]),
        "answer_correctness": safe_float(results["answer_correctness"]),
        "context_precision": safe_float(results["context_precision"]),
        "timestamp": datetime.now(UTC).isoformat(),
        "num_questions": len(TEST_QUESTIONS),
    }

    with open("evaluation_results.json", "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=2)

    return scores