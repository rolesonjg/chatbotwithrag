# Local PDF RAG Demo

A lightweight local RAG demo built with PyMuPDF, ChromaDB, BM25, reciprocal-rank fusion (RRF), LangChain, and Ollama. It includes a basic browser UI for uploading, renaming, deleting, selectively ingesting, and querying PDFs.

## Requirements
- Python 3.10+
- `uv`
- Ollama installed and running
- The configured local model (default: `qwen2.5:1.5b`)

## Setup
Run these commands from this directory:

```bash
uv sync
ollama pull qwen2.5:1.5b
```

If Ollama is already installed and the model is available locally, the pull step is not needed.

## Start the web app

```bash
uv run uvicorn web_app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser. Keep the terminal open while demoing. The API documentation is available at `http://127.0.0.1:8000/docs`.

## UI features
- Upload PDF files (50 MB maximum).
- Per-file status: Uploaded, Ingested, Ingesting, or Failed.
- Ingest only the PDF you choose; ingesting one document preserves other indexed documents.
- Rename a PDF. Renaming an indexed file removes its old index entry; ingest the renamed file to index it again.
- Delete a PDF and its indexed chunks.
- Select ingested PDFs with checkboxes to limit question retrieval. If no checkboxes are selected, all ingested PDFs are searched.
- Ask questions and view structured answers, source document names, page ranges, and an insufficient-evidence warning.
- Reset the database/index without deleting uploaded PDFs.

## Existing CLI tools

```bash
uv run python ingest.py
uv run python rag.py
uv run python inspect_data.py
```

`ingest.py` is the original CLI ingestion command and rebuilds the collection from every PDF in `documents/`. For the browser workflow, use the per-document **Ingest** button instead.

## Data locations
- `documents/`: uploaded PDFs
- `.chroma/`: local vector index
- `data/structured_records.json`: extracted section records
- `data/chunks.json`: chunk text and metadata
- `data/document_status.json`: UI status/error state

## Notes and limitations
- Embedding-model downloads happen the first time the model is loaded; an internet connection may be needed initially.
- Ollama must be running locally. The app does not send your PDFs to a hosted LLM API.
- PDF parsing uses lightweight layout heuristics. Scanned PDFs may need OCR; complex tables and multi-column layouts may not parse perfectly.
- This is a local demo, not a hardened multi-user deployment. Do not expose it to an untrusted network without adding authentication, request limits, and other security controls.
