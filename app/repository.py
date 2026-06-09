from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class RepositoryRef:
    """Normalized identity for a public GitHub repository."""

    repo_id: str
    owner: str
    name: str
    normalized_url: str


@dataclass(frozen=True)
class RepositoryPaths:
    """Repo-scoped runtime paths for cloned source and FAISS index files."""

    repo_id: str
    source_dir: Path
    index_dir: Path


def parse_github_repo_url(repo_url: str) -> RepositoryRef:
    """Validate and normalize a public GitHub repository URL."""
    parsed = urlparse(repo_url.strip())

    if parsed.scheme != "https" or parsed.netloc.lower() != "github.com":
        raise ValueError("Only https://github.com/{owner}/{repo} URLs are supported")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise ValueError("GitHub URL must be in https://github.com/{owner}/{repo} format")

    owner, repo_name = parts
    repo_name = repo_name.removesuffix(".git")

    if not owner or not repo_name:
        raise ValueError("GitHub owner and repository name are required")

    normalized_url = f"https://github.com/{owner}/{repo_name}.git"
    slug = f"{owner}-{repo_name}".lower()
    safe_slug = "".join(char if char.isalnum() else "-" for char in slug).strip("-")
    short_hash = sha1(normalized_url.encode("utf-8")).hexdigest()[:8]

    return RepositoryRef(
        repo_id=f"{safe_slug}-{short_hash}",
        owner=owner,
        name=repo_name,
        normalized_url=normalized_url,
    )


def build_repository_paths(repo_id: str, data_dir: str | Path = "data") -> RepositoryPaths:
    """Build isolated storage paths for one indexed repository."""
    base_dir = Path(data_dir) / "repositories" / repo_id

    return RepositoryPaths(
        repo_id=repo_id,
        source_dir=base_dir / "source",
        index_dir=base_dir / "faiss",
    )


def resolve_repo_file(repo_path: str | Path, file_path: str) -> Path:
    """Resolve a user-provided path and keep it inside the repository root."""
    repo_root = Path(repo_path).resolve()
    candidate = (repo_root / file_path).resolve()

    if candidate == repo_root or repo_root not in candidate.parents:
        raise ValueError("File path must stay inside the indexed repository")

    return candidate