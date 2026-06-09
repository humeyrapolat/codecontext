from dataclasses import dataclass, field
from typing import Any


@dataclass
class RepositoryRuntime:
    repo_id: str
    repo_url: str
    agent_runtime: Any


@dataclass
class AppState:
    """In-memory runtime registry for indexed repositories.

    This is intentionally simple for the prototype: it keeps repo-specific
    AgentRuntime objects available across API requests while the process is
    running. For production, repository metadata and job state should move to
    persistent storage such as Redis or PostgreSQL.
    """
    repositories: dict[str, RepositoryRuntime] = field(default_factory=dict)
    active_repo_id: str | None = None

    @property
    def agent_ready(self) -> bool:
        return self.active_repo_id in self.repositories

    def set_repository(self, repo_id: str, repo_url: str, agent_runtime: Any) -> None:
        self.repositories[repo_id] = RepositoryRuntime(
            repo_id=repo_id,
            repo_url=repo_url,
            agent_runtime=agent_runtime,
        )
        self.active_repo_id = repo_id

    def list_repositories(self) -> list[RepositoryRuntime]:
        return list(self.repositories.values())

    def require_repository(self, repo_id: str | None = None) -> RepositoryRuntime:
        selected_repo_id = repo_id or self.active_repo_id
        if selected_repo_id is None or selected_repo_id not in self.repositories:
            raise RuntimeError("Repository agent is not ready")
        return self.repositories[selected_repo_id]

    def require_agent(self, repo_id: str | None = None) -> Any:
        return self.require_repository(repo_id).agent_runtime

# Shared application state used by FastAPI routes.
# Tests can still instantiate AppState directly, so this avoids a hard singleton class.
app_state = AppState()
