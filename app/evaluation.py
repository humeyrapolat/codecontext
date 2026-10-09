import json
import math
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
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=512,
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

    for test_case in TEST_QUESTIONS:
        question = test_case["question"]
        context_list = get_contexts_for_question(agent_runtime, question)
        context_text = "\n\n".join(context_list)

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
            model="llama-3.1-8b-instant",
            temperature=0,
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