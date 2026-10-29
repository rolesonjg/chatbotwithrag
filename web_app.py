from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import pymupdf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from config import (
    ROOT, DOCUMENTS_DIR, DATA_DIR, CHROMA_DIR, RECORDS_JSON, CHUNKS_JSON,
    COLLECTION, EMBEDDING_MODEL, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
)
from document_parser import parse_pdf
from retriever import retrieve_documents
from rag import ANSWER_PROMPT, RAGAnswer, build_context

app = FastAPI(title="Local PDF RAG Demo")
STATIC_DIR = ROOT / "static"
REGISTRY_PATH = DATA_DIR / "document_status.json"
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_embeddings = None
_store = None

class QuestionRequest(BaseModel):
    question: str
    document_names: list[str] = []

class RenameRequest(BaseModel):
    new_name: str


def embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def store():
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _store


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def registry():
    return read_json(REGISTRY_PATH, {})


def indexed_names():
    try:
        result = store().get(include=["metadatas"])
        return {m.get("document_name") for m in result.get("metadatas", []) if m and m.get("document_name")}
    except Exception:
        return set()


def get_documents():
    indexed = indexed_names()
    status_data = registry()
    output = []
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda p: p.name.lower()):
        try:
            with pymupdf.open(path) as pdf:
                pages = len(pdf)
            status = "Ingested" if path.name in indexed else status_data.get(path.name, {}).get("status", "Uploaded")
            output.append({"name": path.name, "status": status, "pages": pages, "size": path.stat().st_size})
        except Exception:
            output.append({"name": path.name, "status": "Failed", "pages": None, "size": path.stat().st_size})
    return output


def remove_index_for_document(name: str):
    db = store()
    try:
        db.delete(where={"document_name": name})
    except Exception:
        # Compatibility fallback for Chroma versions that don't accept the filter.
        result = db.get(where={"document_name": name}, include=[])
        if result.get("ids"):
            db.delete(ids=result["ids"])
    for path, key in ((RECORDS_JSON, "document_name"), (CHUNKS_JSON, "metadata")):
        items = read_json(path, [])
        kept = []
        for item in items:
            actual = item.get(key) if key == "document_name" else (item.get("metadata") or {}).get("document_name")
            if actual != name:
                kept.append(item)
        write_json(path, kept)


def ingest_one(path: Path):
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files can be ingested.")
    # Validate PDF before expensive parsing/embedding.
    with pymupdf.open(path) as pdf:
        if len(pdf) == 0:
            raise ValueError("This PDF has no pages.")

    records = parse_pdf(path)
    if not records:
        raise ValueError("No readable text was found. Scanned PDFs may need OCR.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for record in records:
        metadata = {k: record[k] for k in (
            "document_id", "document_name", "source", "section_title",
            "section_path", "page_start", "page_end", "record_type"
        )}
        chunks.extend(splitter.split_documents([Document(page_content=record["content"], metadata=metadata)]))

    for chunk in chunks:
        raw = "|".join(str(chunk.metadata.get(k, "")) for k in (
            "document_id", "page_start", "section_path", "start_index"
        )) + "|" + chunk.page_content
        chunk.metadata["chunk_id"] = hashlib.sha256(raw.encode()).hexdigest()

    # Replace this document's old index only; keep other ingested PDFs intact.
    remove_index_for_document(path.name)
    db = store()
    batch_size = 12
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        db.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])

    old_records = [r for r in read_json(RECORDS_JSON, []) if r.get("document_name") != path.name]
    old_records.extend(records)
    write_json(RECORDS_JSON, old_records)

    old_chunks = [c for c in read_json(CHUNKS_JSON, []) if (c.get("metadata") or {}).get("document_name") != path.name]
    old_chunks.extend([{"page_content": c.page_content, "metadata": c.metadata} for c in chunks])
    write_json(CHUNKS_JSON, old_chunks)

    status_data = registry()
    status_data[path.name] = {"status": "Ingested", "error": None}
    write_json(REGISTRY_PATH, status_data)
    return {"document": path.name, "records": len(records), "chunks": len(chunks)}


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/documents")
def list_documents():
    return {"documents": get_documents(), "indexed_chunks": len(store().get(include=[]).get("ids", []))}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    # Prevent overwriting an existing document silently.
    target = DOCUMENTS_DIR / filename
    if target.exists():
        raise HTTPException(status_code=409, detail="A document with that filename already exists. Rename it first or delete the old file.")
    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The uploaded file does not look like a valid PDF.")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum upload size is 50 MB.")
    target.write_bytes(content)
    statuses = registry()
    statuses[filename] = {"status": "Uploaded", "error": None}
    write_json(REGISTRY_PATH, statuses)
    return {"message": "PDF uploaded. Ingest it when ready.", "document": filename}


@app.post("/api/documents/{name}/ingest")
def ingest_document(name: str):
    safe_name = Path(name).name
    path = DOCUMENTS_DIR / safe_name
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=404, detail="PDF not found.")
    statuses = registry()
    statuses[safe_name] = {"status": "Ingesting", "error": None}
    write_json(REGISTRY_PATH, statuses)
    try:
        result = ingest_one(path)
        return {"message": "Ingestion completed.", **result}
    except Exception as exc:
        statuses = registry()
        statuses[safe_name] = {"status": "Failed", "error": str(exc)[:500]}
        write_json(REGISTRY_PATH, statuses)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.patch("/api/documents/{name}")
def rename_document(name: str, request: RenameRequest):
    old_name = Path(name).name
    old_path = DOCUMENTS_DIR / old_name
    new_name = Path(request.new_name).name.strip()
    if not old_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    if not new_name.lower().endswith(".pdf") or not new_name or new_name in {".pdf", "..pdf"}:
        raise HTTPException(status_code=400, detail="The new filename must end in .pdf.")
    new_path = DOCUMENTS_DIR / new_name
    if new_path.exists():
        raise HTTPException(status_code=409, detail="A file with that name already exists.")
    was_indexed = old_name in indexed_names()
    if was_indexed:
        remove_index_for_document(old_name)
    old_path.rename(new_path)
    statuses = registry()
    statuses.pop(old_name, None)
    statuses[new_name] = {"status": "Uploaded", "error": None}
    write_json(REGISTRY_PATH, statuses)
    return {"message": "Document renamed. Re-ingest it to index the renamed document.", "was_indexed": was_indexed}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    safe_name = Path(name).name
    path = DOCUMENTS_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")
    remove_index_for_document(safe_name)
    path.unlink()
    statuses = registry()
    statuses.pop(safe_name, None)
    write_json(REGISTRY_PATH, statuses)
    return {"message": f"Deleted {safe_name} and its indexed chunks."}


@app.post("/api/reset")
def reset_database():
    db = store()
    existing = db.get(include=[]).get("ids", [])
    if existing:
        db.delete(ids=existing)
    write_json(RECORDS_JSON, [])
    write_json(CHUNKS_JSON, [])
    statuses = registry()
    for item in get_documents():
        statuses[item["name"]] = {"status": "Uploaded", "error": None}
    write_json(REGISTRY_PATH, statuses)
    return {"message": "Database reset. PDF files were kept; ingest the documents you want to use.", "deleted_chunks": len(existing)}


@app.post("/api/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Enter a question first.")
    selected = [Path(name).name for name in request.document_names]
    indexed = indexed_names()
    if selected:
        selected = [name for name in selected if name in indexed]
        if not selected:
            raise HTTPException(status_code=400, detail="Select at least one ingested document.")
    elif not indexed:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")

    try:
        results = retrieve_documents(question, TOP_K, document_names=selected or None)
        if not results:
            return {"answer": "I couldn't find relevant passages in the selected documents.", "key_points": [], "sources": [], "context_sufficient": False, "warning": "No relevant passages were retrieved."}
        context = build_context(results)
        model = ChatOllama(model=OLLAMA_MODEL, temperature=0)
        answer_model = model.with_structured_output(RAGAnswer, method="json_schema")
        messages = ANSWER_PROMPT.format_messages(question=question, context=context)
        answer = answer_model.invoke(messages)
        payload = answer.model_dump()
        # Attach deterministic sources from retrieved metadata, not model-invented sources.
        payload["sources"] = []
        seen = set()
        for doc, score in results:
            md = doc.metadata
            source = {
                "document": md.get("document_name", "Unknown document"),
                "page_start": md.get("page_start"),
                "page_end": md.get("page_end"),
                "section": md.get("section_path", ""),
            }
            key = (source["document"], source["page_start"], source["page_end"])
            if key not in seen:
                seen.add(key)
                payload["sources"].append(source)
        if not payload.get("context_sufficient"):
            payload["warning"] = "The retrieved passages may not contain enough evidence for a complete answer."
        return payload
    except Exception as exc:
        message = str(exc)
        if "Connection refused" in message or "connect" in message.lower() and "ollama" in message.lower():
            raise HTTPException(status_code=503, detail="Cannot connect to Ollama. Start Ollama and ensure the configured model is installed.")
        raise HTTPException(status_code=500, detail=f"Could not answer the question: {message[:400]}")


@app.get("/api/health")
def health():
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            ollama_ok = response.status == 200
    except Exception:
        ollama_ok = False
    return {"ok": True, "ollama": ollama_ok, "model": OLLAMA_MODEL}
