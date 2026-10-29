import hashlib
import json
from config import DOCUMENTS_DIR, DATA_DIR, CHROMA_DIR, RECORDS_JSON, CHUNKS_JSON, COLLECTION, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from document_parser import parse_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"Put at least one PDF in {DOCUMENTS_DIR}")
        return

    records = []
    for pdf in pdfs:
        print(f"Parsing {pdf.name}...", flush=True)
        extracted = parse_pdf(pdf)
        print(f"  Found {len(extracted)} section/body records")
        records.extend(extracted)

    save_json(RECORDS_JSON, records)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for r in records:
        md = {k: r[k] for k in (
            "document_id", "document_name", "source", "section_title",
            "section_path", "page_start", "page_end", "record_type"
        )}
        chunks.extend(splitter.split_documents([Document(page_content=r["content"], metadata=md)]))

    for c in chunks:
        raw = "|".join(str(c.metadata.get(k, "")) for k in (
            "document_id", "page_start", "section_path", "start_index"
        )) + "|" + c.page_content
        c.metadata["chunk_id"] = hashlib.sha256(raw.encode()).hexdigest()

    save_json(CHUNKS_JSON, [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks])
    print(f"Chunks created: {len(chunks)}")

    print("Loading embedding model...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))

    # This project owns this collection; rebuild it to avoid stale records.
    old = store.get(include=[])
    if old.get("ids"):
        store.delete(ids=old["ids"])
        print(f"Deleted old chunks: {len(old['ids'])}")

    batch_size = 12
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
        print(f"Indexed {min(start + batch_size, len(chunks))}/{len(chunks)}", flush=True)
    print(f"Done. Chroma contains {len(store.get(include=[])['ids'])} chunks.")
    print(f"Review {RECORDS_JSON} and {CHUNKS_JSON} before evaluating retrieval.")


if __name__ == "__main__":
    main()
