import os
import git
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Desteklenen kod uzantıları ve dilleri
# Neden bu liste? Agent sadece kod dosyalarına bakmalı
# README, resim, lock dosyaları indexlememeli — gürültü olur
SUPPORTED_EXTENSIONS = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.JS,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".md": None,  # Markdown düz metin olarak işle
}

# Embedding modeli — LearnFlow'dan tanıdık
# Bir kere yükle, hep kullan — her istekte yeniden yükleme = yavaş
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def clone_repo(repo_url: str, target_dir: str = "indexed_repo") -> str:
    """
    GitHub reposunu local'e indir.
    
    Neden clone? Repoyu dosya dosya okumak için local'de olması lazım.
    git.Repo.clone_from() → GitPython kütüphanesi, git komutlarını Python'dan çalıştırır.
    """
    # Zaten indirilmişse tekrar indirme — hem yavaş hem gereksiz
    if os.path.exists(target_dir):
        print(f"✅ Repo zaten mevcut: {target_dir}")
        return target_dir
    
    print(f"📥 Repo indiriliyor: {repo_url}")
    git.Repo.clone_from(repo_url, target_dir)
    print(f"✅ Repo indirildi: {target_dir}")
    return target_dir


def load_code_files(repo_dir: str) -> list[Document]:
    """
    Repo içindeki kod dosyalarını oku, Document objelerine çevir.
    
    Neden Document? LangChain'in standart formatı.
    İçinde page_content (kod) ve metadata (dosya yolu, dil) var.
    Metadata önemli — agent "bu kod hangi dosyadan?" diye sorabilir.
    """
    documents = []
    repo_path = Path(repo_dir)
    
    for file_path in repo_path.rglob("*"):  # tüm dosyaları recursive tara
        
        # Sadece desteklenen uzantıları al
        if file_path.suffix not in SUPPORTED_EXTENSIONS:
            continue
            
        # Gizli klasörleri atla (.git, .venv, node_modules vs.)
        # Neden? .git klasöründe binary dosyalar var, okunmaz
        # node_modules'da binlerce dosya var, gürültü olur
        if any(part.startswith(".") or part in ["node_modules", "__pycache__", ".venv"] 
               for part in file_path.parts):
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            
            # Boş dosyaları atla — indexlemenin anlamı yok
            if not content.strip():
                continue
                
            # Çok büyük dosyaları atla (100KB+) — genellikle generated code
            if len(content) > 100_000:
                continue
            
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "language": SUPPORTED_EXTENSIONS[file_path.suffix],
                    "filename": file_path.name,
                    # Relative path — repo içindeki konumu
                    "relative_path": str(file_path.relative_to(repo_path))
                }
            ))
            
        except Exception as e:
            print(f"⚠️ Dosya okunamadı: {file_path} — {e}")
            continue
    
    print(f"📄 Toplam {len(documents)} kod dosyası okundu")
    return documents


def chunk_code(documents: list[Document]) -> list[Document]:
    """
    Kod dosyalarını akıllıca böl.
    
    Neden RecursiveCharacterTextSplitter(Language.PYTHON)?
    Bu splitter Python syntax'ını anlıyor:
    - Önce class sınırlarında bölmeye çalışır
    - Sonra fonksiyon sınırlarında
    - Sonra satır sınırlarında
    - En son karakter sınırında
    
    Yani fonksiyon ortasında kesmekten kaçınıyor — çok önemli!
    """
    all_chunks = []
    
    for doc in documents:
        language = doc.metadata.get("language")
        
        if language is not None:
            # Dile özel splitter — syntax'ı anlıyor
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1000,   # Kod için 1000 karakter — PDF'den büyük
                chunk_overlap=100  # Fonksiyonlar arası bağlam için
            )
        else:
            # Markdown için düz splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
        
        chunks = splitter.split_documents([doc])
        
        # Her chunk'a kaynak dosya bilgisini koru
        # Splitter metadata'yı taşıyor ama emin olmak için
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        
        all_chunks.extend(chunks)
    
    print(f"✂️ Toplam {len(all_chunks)} chunk oluşturuldu")
    return all_chunks


def build_index(repo_url: str, index_dir: str = "faiss_index") -> FAISS:
    """
    Tam pipeline:
    GitHub URL → clone → dosyaları oku → chunk → embed → FAISS'e kaydet
    
    Bu fonksiyonu API endpoint'inden çağıracağız.
    """
    # 1. Repoyu indir
    repo_dir = clone_repo(repo_url)
    
    # 2. Dosyaları oku
    documents = load_code_files(repo_dir)
    
    if not documents:
        raise ValueError("Hiç kod dosyası bulunamadı!")
    
    # 3. Chunk'la
    chunks = chunk_code(documents)
    
    # 4. Embed et ve FAISS'e kaydet
    print("🔄 Embedding başlıyor...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    print(f"✅ Index oluşturuldu: {index_dir}")
    
    return vectorstore


def load_index(index_dir: str = "faiss_index") -> FAISS:
    """
    Kaydedilmiş FAISS index'i yükle.
    Her soruda tekrar embedding yapma — hem yavaş hem paralı.
    """
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )