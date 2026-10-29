from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / ".chroma"
RECORDS_JSON = DATA_DIR / "structured_records.json"
CHUNKS_JSON = DATA_DIR / "chunks.json"
COLLECTION = "general_pdf_rag"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:1.5b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 140
TOP_K = 5
CANDIDATE_K = 12
