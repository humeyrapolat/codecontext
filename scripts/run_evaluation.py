"""Run the RAGAS evaluation end-to-end against CodeContext's own repository.

This needs real network access to the Groq API, which the hosted Claude
sandbox this script was written in cannot reach (org egress policy). Run it
yourself, locally, with GROQ_API_KEY set in .env:

    uv run python scripts/run_evaluation.py

First run clones+indexes https://github.com/humeyrapolat/codecontext into
./indexed_repo and ./faiss_index (gitignored), then answers and scores the
30-question set in app/evaluation.py. Results are written to
evaluation_results.json (also gitignored) and printed to stdout.
"""

from dotenv import load_dotenv

from app.agent import build_agent
from app.indexer import build_index
from app.evaluation import run_evaluation

REPO_URL = "https://github.com/humeyrapolat/codecontext.git"


def main() -> None:
    load_dotenv()

    print(f"Indexing {REPO_URL} ...")
    index_result = build_index(REPO_URL)
    print(
        f"Indexed {index_result.documents_count} files "
        f"into {index_result.chunks_count} chunks."
    )

    agent_runtime = build_agent(
        index_result.vectorstore,
        str(index_result.repo_path),
        repo_id="self-eval",
    )

    print("Running evaluation (this calls the Groq API once per question)...")
    scores = run_evaluation(agent_runtime)

    print("\nResults:")
    for key, value in scores.items():
        print(f"  {key}: {value}")
    print("\nWritten to evaluation_results.json")


if __name__ == "__main__":
    main()
