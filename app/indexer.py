import os
from dataclasses import dataclass
from pathlib import Path

import git
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "js",
    ".ts": "js",
    ".java": "java",
    ".kt": "kotlin",
    ".md": None,
}

IGNORED_PATH_PARTS = {"node_modules", "__pycache__", ".venv"}
MAX_FILE_SIZE_CHARS = 100_000

_embeddings = None


@dataclass
class IndexResult:
    vectorstore: FAISS
    repo_path: Path
    index_path: Path
    documents_count: int
    chunks_count: int


def get_embeddings():
    """Load the embedding model lazily to keep API startup lightweight."""
    global _embeddings

    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return _embeddings


def build_text_splitter(language: str | None):
    """Create a language-aware splitter only when indexing is needed."""
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

    if language is None:
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

    language_map = {
        "python": Language.PYTHON,
        "js": Language.JS,
        "java": Language.JAVA,
        "kotlin": Language.KOTLIN,
    }

    return RecursiveCharacterTextSplitter.from_language(
        language=language_map[language],
        chunk_size=1000,
        chunk_overlap=100,
    )


def clone_repo(repo_url: str, target_dir: str | Path = "indexed_repo") -> Path:
    """Clone a public GitHub repository into the target directory."""
    target_path = Path(target_dir)

    if target_path.exists():
        print(f"Repository already exists: {target_path}")
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    git.Repo.clone_from(repo_url, target_path)

    return target_path


def should_skip_file(file_path: Path) -> bool:
    if file_path.suffix not in SUPPORTED_EXTENSIONS:
        return True

    return any(
        part.startswith(".") or part in IGNORED_PATH_PARTS
        for part in file_path.parts
    )


def load_code_files(repo_dir: str | Path) -> list[Document]:
    """Load supported source and Markdown files into LangChain documents."""
    documents = []
    repo_path = Path(repo_dir)

    for file_path in repo_path.rglob("*"):
        if should_skip_file(file_path):
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"Could not read file {file_path}: {exc}")
            continue

        if not content.strip():
            continue

        if len(content) > MAX_FILE_SIZE_CHARS:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "language": SUPPORTED_EXTENSIONS[file_path.suffix],
                    "filename": file_path.name,
                    "relative_path": str(file_path.relative_to(repo_path)),
                },
            )
        )

    return documents


def chunk_code(documents: list[Document]) -> list[Document]:
    """Split source documents into retrieval-friendly chunks."""
    all_chunks = []

    for document in documents:
        language = document.metadata.get("language")
        splitter = build_text_splitter(language)
        chunks = splitter.split_documents([document])

        for chunk in chunks:
            chunk.metadata.update(document.metadata)

        all_chunks.extend(chunks)

    return all_chunks


def build_index(
    repo_url: str,
    repo_dir: str | Path = "indexed_repo",
    index_dir: str | Path = "faiss_index",
) -> IndexResult:
    """Build a FAISS index from a public GitHub repository."""
    repo_path = clone_repo(repo_url, repo_dir)
    documents = load_code_files(repo_path)

    if not documents:
        raise ValueError("No supported code or Markdown files were found")

    chunks = chunk_code(documents)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    return IndexResult(
        vectorstore=vectorstore,
        repo_path=repo_path,
        index_path=Path(index_dir),
        documents_count=len(documents),
        chunks_count=len(chunks),
    )


def load_index(index_dir: str | Path = "faiss_index") -> FAISS:
    """Load a persisted FAISS index.

    FAISS uses pickle-based metadata serialization. Only load indexes created
    by this application from trusted local paths.
    """
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )