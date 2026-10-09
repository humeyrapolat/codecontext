# CodeContext CV Snippet

## English CV Version

**CodeContext - AI Codebase Assistant**  
Python, FastAPI, LangChain, FAISS, HuggingFace Embeddings, Groq LLaMA 3.3 70B, LangFuse, RAGAS, GitPython, uv

- Built an AI-powered codebase assistant that indexes public GitHub repositories and answers natural-language questions about source code using RAG and LangChain tool-calling agents.
- Implemented a repository indexing pipeline with Git cloning, file filtering, language-aware chunking, local HuggingFace embeddings, and FAISS vector index creation.
- Refactored the prototype into a repo-scoped runtime architecture with stable `repo_id` values, isolated storage paths, per-repository FAISS indexes, and an `AgentRuntime` abstraction.
- Built agent tools for semantic code search, repository file listing, and safe file content lookup with path traversal protection.
- Added repo-scoped conversation memory so multi-turn sessions stay isolated by repository and session id.
- Integrated LangFuse tracing for LLM pipeline observability, including retrieval, tool-calling, latency, and token-level debugging.
- Added a RAGAS evaluation entrypoint for measuring faithfulness, answer correctness, and context precision without publishing unsupported benchmark claims.
- Added unit tests and GitHub Actions CI for repository parsing, path safety, runtime state behavior, and Python compilation checks.

## Short CV Version

**CodeContext - AI Codebase Assistant**  
Built a FastAPI and LangChain-based assistant that indexes GitHub repositories, creates FAISS vector indexes with local HuggingFace embeddings, and answers codebase questions through repo-scoped tool-calling agents. Added safe file tools, URL validation, repo-scoped memory, LangFuse tracing, RAGAS evaluation support, unit tests, and GitHub Actions CI.

## Turkish Explanation

CodeContext, public GitHub repository'lerini indexleyip kod hakkında doğal dilde soru cevaplayabilen bir AI codebase assistant projesidir. Projede RAG pipeline, FAISS vector store, LangChain tool-calling agent, repo bazlı runtime state, güvenli dosya okuma, LangFuse observability, RAGAS evaluation desteği, test ve CI altyapısı bulunmaktadır.

## Interview Pitch

I started with a working RAG prototype and focused on turning it into a more production-oriented AI application. The main architectural improvement was moving from a single global active agent to repo-scoped runtime state. Each indexed repository gets a stable `repo_id`, isolated storage paths, its own FAISS index, and an `AgentRuntime` that owns the tools and vectorstore used to answer questions. This makes the system easier to reason about, safer to extend, and ready for future multi-repository and persistent-state support.

## Technical Talking Points

- RAG pipeline design: clone, filter, chunk, embed, retrieve, and answer.
- Why FAISS was chosen for local-first vector search.
- Why local HuggingFace embeddings reduce indexing API cost.
- Why global state is risky in multi-user or multi-repository systems.
- Why tools should be bound to a specific repository and vectorstore.
- How path traversal protection prevents file tools from reading outside the indexed repo.
- How LangFuse tracing helps debug retrieval, tool use, latency, and token usage.
- How RAGAS can evaluate faithfulness, context precision, and answer correctness.
- What would be needed next for production: background jobs, persistent state, citations, reranking, streaming, and private repo support.
