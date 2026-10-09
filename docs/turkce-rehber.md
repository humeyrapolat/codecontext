# CodeContext Türkçe Rehber

Bu doküman CodeContext projesini hiç bilmeyen biri için açıklamak amacıyla hazırlanmıştır. Amaç sadece kodun ne yaptığını değil, projenin arkasındaki backend, RAG ve agent mimarisini de anlaşılır hale getirmektir.

## CodeContext Nedir?

CodeContext, public GitHub repository'lerini indexleyip o kod tabanı hakkında doğal dilde soru cevaplayabilen bir AI codebase assistant projesidir.

Kısaca:

```text
GitHub repo URL'i verilir
→ repo local'e clone edilir
→ kod dosyaları okunur
→ kodlar küçük parçalara bölünür
→ embedding üretilir
→ FAISS vector index oluşturulur
→ kullanıcı soru sorar
→ agent ilgili kod parçalarını tool'larla bulur
→ LLM cevap üretir
```

Yani proje şu fikri uygular:

```text
"Bir GitHub repo ile sohbet et."
```

## Hangi Problemi Çözüyor?

Yeni bir codebase'e girdiğinde şu sorular genelde zaman alır:

- Proje nasıl çalışıyor?
- API endpointleri nerede?
- Agent logic hangi dosyada?
- Repository indexing nasıl yapılıyor?
- Hangi dosya hangi sorumluluğa sahip?
- Bu fonksiyon veya class ne işe yarıyor?

CodeContext bu süreci hızlandırmak için kod tabanını semantik olarak aranabilir hale getirir ve kullanıcının sorularına ilgili kod parçalarına dayanarak cevap verir.

## Temel Kavramlar

### RAG Nedir?

RAG, Retrieval-Augmented Generation demektir.

Basitçe:

```text
Önce ilgili bilgiyi bul
→ sonra LLM'e bu bilgiyle cevap üret
```

CodeContext'te RAG şu şekilde çalışır:

```text
Kullanıcı soru sorar
→ FAISS en alakalı kod chunk'larını bulur
→ bu context LLM'e verilir
→ LLM cevap üretir
```

Bu yaklaşım LLM'in sadece kendi ezberine dayanmasını azaltır. Cevaplar codebase içindeki gerçek dosyalara ve kod parçalarına daha yakın olur.

### Embedding Nedir?

Embedding, bir metni sayısal vektöre dönüştürmektir.

Örneğin:

```text
"FastAPI endpointleri nerede?"
```

ve

```text
"API routes hangi dosyada tanımlanıyor?"
```

kelime olarak farklıdır ama anlam olarak yakındır. Embedding sayesinde sistem bu anlam yakınlığını yakalayabilir.

CodeContext local HuggingFace embedding modeli kullanır:

```text
all-MiniLM-L6-v2
```

### FAISS Nedir?

FAISS, embedding vektörleri içinde hızlı arama yapmayı sağlayan vector search kütüphanesidir.

CodeContext'te FAISS şu işe yarar:

```text
Soru embedding'e çevrilir
→ FAISS en benzer kod chunk'larını bulur
→ agent/LLM bu chunk'lara göre cevap verir
```

### Agent Nedir?

Bu projede agent, LLM'in sadece cevap yazmadığı, aynı zamanda tool kullanabildiği yapıdır.

CodeContext agent'ının tool'ları:

- `search_code`: Kod içinde semantik arama yapar.
- `list_files`: Repository dosya yapısını listeler.
- `get_file_content`: Belirli bir dosyanın içeriğini okur.

Örnek:

```text
Kullanıcı: "Repository indexing nasıl çalışıyor?"
Agent:
1. search_code tool'unu çağırır
2. indexer.py ile ilgili chunk'ları bulur
3. gerekirse get_file_content ile dosyayı okur
4. LLM ile açıklama üretir
```

## Proje Mimarisi

Genel mimari:

```text
GitHub URL
  → URL validation
  → repo_id generation
  → repo-scoped storage paths
  → Git clone
  → file filtering
  → language-aware chunking
  → local embeddings
  → FAISS index
  → AgentRuntime
  → FastAPI endpoints
  → LangChain tool calls
  → Groq LLM answer
```

## Klasör Yapısı

```text
codecontext/
  app/
    api.py
    agent.py
    indexer.py
    repository.py
    state.py
    evaluation.py
  tests/
    test_repository.py
    test_state.py
  docs/
    cv-codecontext.md
    turkce-rehber.md
  .github/workflows/ci.yml
  .env.example
  pyproject.toml
  uv.lock
```

## Ana Dosyalar

### app/api.py

FastAPI endpointleri burada tanımlanır.

Önemli endpointler:

```text
POST /repositories
GET /repositories
POST /repositories/{repo_id}/ask
POST /ask
POST /evaluate
GET /status
```

Bu dosyanın görevi:

```text
HTTP request al
→ doğru servisi çağır
→ response modeliyle cevap dön
```

Senior bakış:

```text
api.py işin tamamını yapmaz.
Indexing'i indexer.py'ye, agent logic'i agent.py'ye, state yönetimini state.py'ye bırakır.
```

### app/indexer.py

Repository'yi AI'ın arayabileceği hale getirir.

Pipeline:

```text
clone_repo
→ load_code_files
→ chunk_code
→ FAISS.from_documents
→ vectorstore.save_local
```

Bu dosya şu sorumluluklara sahiptir:

- GitHub repo'yu clone etmek
- Desteklenen dosya türlerini seçmek
- Gereksiz klasörleri filtrelemek
- Kod dosyalarını LangChain `Document` objelerine çevirmek
- Kodları chunk'lara bölmek
- Embedding ve FAISS index oluşturmak

### app/agent.py

Tool-calling agent logic burada bulunur.

Ana kavram:

```text
AgentRuntime
```

Her repository için ayrı bir `AgentRuntime` oluşturulur. Bu runtime şunları içerir:

- LLM + tools
- Tool listesi
- FAISS vectorstore
- Repository path
- repo_id

Bu sayede her repo kendi agent'ına, kendi tool'larına ve kendi vectorstore'una sahip olur.

### app/repository.py

Repository kimliği ve path güvenliği burada yönetilir.

Yaptığı işler:

- GitHub URL doğrulama
- URL'i normalize etme
- stable `repo_id` üretme
- repo'ya özel storage path oluşturma
- file tool'ları için path traversal koruması

Örnek:

```text
https://github.com/humeyrapolat/codecontext
```

normalize edilir:

```text
https://github.com/humeyrapolat/codecontext.git
```

ve bir `repo_id` üretilir:

```text
humeyrapolat-codecontext-<hash>
```

### app/state.py

Runtime state yönetimi burada yapılır.

Proje birden fazla repository indexleyebilir. Sistem şunu bilmek zorundadır:

- Hangi repository'ler indexlendi?
- Aktif repository hangisi?
- Hangi repo hangi agent runtime'a sahip?

`AppState` bu bilgileri memory'de tutar.

Basit model:

```text
AppState
  repositories:
    repo_id → RepositoryRuntime
  active_repo_id:
    son indexlenen repo
```

Bu bir module-level shared state yaklaşımıdır:

```python
app_state = AppState()
```

Bu klasik hard singleton değildir. Testlerde ayrı `AppState()` instance'ı oluşturulabilir.

Production için bu yaklaşımın sınırı vardır:

- Uygulama restart olursa state kaybolur.
- Birden fazla worker varsa state paylaşılmaz.
- Daha ileri aşamada Redis veya PostgreSQL'e taşınmalıdır.

### app/evaluation.py

RAGAS evaluation entrypoint burada bulunur.

Bu dosya küçük bir evaluation smoke test sağlar:

- FAISS'ten context alır
- LLM ile answer üretir
- RAGAS metriklerini çalıştırır
- `evaluation_results.json` dosyasına sonuç yazar

Ölçülen metrikler:

- faithfulness
- answer correctness
- context precision

Önemli not:

```text
Bu evaluation küçük bir entrypoint'tir.
CV'de benchmark skoru yazmak için daha büyük ve tekrarlanabilir bir evaluation dataset gerekir.
```

## Repo-Scoped Architecture Nedir?

İlk prototiplerde sık yapılan hata şudur:

```text
tek global vectorstore
tek global agent
tek aktif repo
```

Bu yaklaşım birden fazla repository için risklidir.

CodeContext bunu şu yapıya taşır:

```text
repo A
  → repo A source path
  → repo A FAISS index
  → repo A AgentRuntime

repo B
  → repo B source path
  → repo B FAISS index
  → repo B AgentRuntime
```

Böylece:

```text
repo A sorusu repo A agent'ına gider
repo B sorusu repo B agent'ına gider
```

Bu mimari CV'de güçlü görünür çünkü sadece RAG değil, runtime isolation düşünülmüştür.

## Path Traversal Protection

Agent'ın file-reading tool'u dosya okuyabilir. Bu güçlü ama riskli bir özelliktir.

Kötü niyetli veya yanlış bir input şöyle olabilir:

```text
../../../.env
```

Bu yüzden `resolve_repo_file` fonksiyonu dosya yolunu repository root içinde kalmaya zorlar.

Amaç:

```text
Agent sadece indexed repository içindeki dosyaları okuyabilsin.
```

## Conversation Memory

Agent, session bazlı konuşma geçmişi tutar.

Session key şu şekilde üretilir:

```text
repo_id:session_id
```

Bu sayede aynı session id farklı repository'lerde kullanılsa bile geçmiş karışmaz.

Örnek:

```text
repo-1:default
repo-2:default
```

Bu iki konuşma ayrı tutulur.

## API Akışı

### 1. Repository Indexleme

Endpoint:

```text
POST /repositories
```

Request:

```json
{
  "repo_url": "https://github.com/humeyrapolat/codecontext"
}
```

Akış:

```text
api.py
→ parse_github_repo_url
→ build_repository_paths
→ build_index
→ build_agent
→ app_state.set_repository
→ IndexResponse
```

### 2. Soru Sorma

Endpoint:

```text
POST /repositories/{repo_id}/ask
```

Request:

```json
{
  "question": "How does repository indexing work?",
  "session_id": "demo-session"
}
```

Akış:

```text
api.py
→ app_state.require_repository(repo_id)
→ ask_agent
→ tool-calling loop
→ QuestionResponse
```

## Testler

Projede lightweight unit testler vardır.

Çalıştırmak için:

```bash
python3 -m unittest discover -s tests
python3 -m compileall app tests main.py
```

Test edilen konular:

- GitHub URL parsing
- repo_id üretimi
- repo-scoped path oluşturma
- path traversal engelleme
- AppState davranışı
- aktif repo yönetimi
- multi-repository state

## CI

GitHub Actions CI şunları çalıştırır:

```text
uv sync --frozen
python -m compileall app main.py
python -m unittest discover -s tests
```

Bu, GitHub'a push veya pull request geldiğinde temel kalite kontrolü sağlar.

## Mevcut Sınırlamalar

Bu proje production-ready değildir; production-oriented portfolio projesidir.

Mevcut sınırlamalar:

- Runtime state memory'dedir.
- Uygulama restart olursa indexlenen repo state'i kaybolur.
- Indexing request lifecycle içinde yapılır.
- Background job sistemi yoktur.
- Private repo desteği yoktur.
- Cevaplarda henüz citation line range yoktur.
- Evaluation dataset küçüktür.

## Gelecek İyileştirmeler

Önerilen roadmap:

```text
1. Background indexing jobs
2. Job status endpoint
3. Persistent state with Redis or PostgreSQL
4. Source citations with file paths and line ranges
5. Hybrid retrieval and reranking
6. Private GitHub repo support
7. Streaming answers
```

## CV'de Nasıl Anlatılır?

Kısa versiyon:

```text
Built a FastAPI and LangChain-based AI codebase assistant that indexes public GitHub repositories, creates FAISS vector indexes with local HuggingFace embeddings, and answers codebase questions through repo-scoped tool-calling agents.
```

Daha teknik versiyon:

```text
Refactored a RAG prototype into a repo-scoped codebase assistant architecture with stable repo_id generation, isolated storage paths, per-repository AgentRuntime instances, semantic code search tools, safe file access, LangFuse observability, RAGAS evaluation support, unit tests, and GitHub Actions CI.
```

