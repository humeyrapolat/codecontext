from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from datasets import Dataset
from app.agent import _vectorstore
from dotenv import load_dotenv
import json
import math
from datetime import datetime
from ragas.metrics import faithfulness, context_precision, answer_correctness # answer_relevancy yerine answer_correctness

load_dotenv()


TEST_QUESTIONS = [
   
    {
        "question": "Which vector store is used?",
        "ground_truth": "FAISS vector store is used."
    },
    {
        "question": "How does the ask endpoint work?",
        "ground_truth": "It receives a question and uses RAG chain to find relevant chunks and generate an answer."
    },
    {
        "question": "What is the API framework used?",
        "ground_truth": "FastAPI is used as the API framework."
    }
]


def get_contexts_for_question(question: str) -> list[str]:
    if _vectorstore is None:
        return ["No vectorstore available"]
    docs = _vectorstore.similarity_search(question, k=3)
    return [doc.page_content for doc in docs]


def safe_float(val):
    if isinstance(val, list):
        v = float(val[0]) if val else 0.0
    else:
        v = float(val)
    return round(0.0 if math.isnan(v) else v, 3)


def run_evaluation(agent_tuple) -> dict:
    print("🔄 Evaluation başlıyor...")
    print(f"📝 {len(TEST_QUESTIONS)} soru test edilecek\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    plain_llm = ChatGroq(
        model="llama-3.1-8b-instant", # Using the smaller model to save tokens
        temperature=0,
        max_tokens=512,
    )

    for i, test in enumerate(TEST_QUESTIONS):
        print(f"[{i+1}/{len(TEST_QUESTIONS)}] Soru: {test['question'][:50]}...")

        # FAISS'ten context'leri al
        context_list = get_contexts_for_question(test["question"])
        context_text = "\n\n".join(context_list)

        messages = [
            SystemMessage(content="""You are a code assistant. 
Answer the question using the provided context. 
Give a detailed and complete answer explaining how things work."""),
            HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {test['question']}\n\nAnswer:")
        ]

        try:
            response = plain_llm.invoke(messages)
            answer = response.content
        except Exception as e:
            print(f"   ⚠️ Hata: {e}")
            answer = "Could not generate answer."

        questions.append(test["question"])
        answers.append(answer)
        contexts.append(context_list)
        ground_truths.append(test["ground_truth"])

        print(f"   ✅ Cevap alındı ({len(answer)} karakter)")

    print("\n🔄 RAGAS metrikleri hesaplanıyor...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    ragas_llm = LangchainLLMWrapper(ChatGroq(
        model="llama-3.1-8b-instant", # Switch to Mixtral for the evaluator
        temperature=0,
        # REMOVE max_tokens here entirely. RAGAS needs space to output complex JSON structures.
    ))

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
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(TEST_QUESTIONS)
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(scores, f, indent=2)

    print("\n✅ Evaluation tamamlandı!")
    print(f"📊 Faithfulness:      {scores['faithfulness']}")
    print(f"📊 Answer Relevancy:  {scores['answer_correctness']}")
    print(f"📊 Context Precision: {scores['context_precision']}")

    return scores