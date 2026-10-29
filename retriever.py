import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    COLLECTION,
    EMBEDDING_MODEL,
    TOP_K,
    CANDIDATE_K,
)


def get_vector_store():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        ),
        persist_directory=str(CHROMA_DIR),
    )


def tokens(text):
    return re.findall(r"\w+", text.lower())


def expand_query(query):
    """
    Add useful alternative terms for common document questions.
    This is rule-based expansion, so it does not require another LLM call.
    """
    q = query.lower().strip()
    queries = [query]

    if "smart" in q:
        queries.extend([
            "SMART specific measurable achievable realistic timely",
            "recommendations should usually be SMART",
        ])

    if "executive summary" in q:
        queries.append(
            "Executive Summary purpose overall report findings objectives"
        )

    if "conclusion" in q:
        queries.append(
            "Conclusion consolidates research findings problems issues"
        )

    if any(term in q for term in ("what does", "what is", "define", "meaning of")):
        queries.append(query + " definition meaning explanation")

    # Remove duplicate queries while preserving order.
    return list(dict.fromkeys(queries))


def document_key(doc):
    return doc.metadata.get("chunk_id") or (
        doc.metadata.get("source"),
        doc.metadata.get("page_start"),
        doc.metadata.get("start_index"),
        doc.page_content,
    )


def rrf(lists, k=TOP_K, rrf_k=60):
    scores = {}
    docs = {}

    for result_list in lists:
        seen = set()

        for rank, doc in enumerate(result_list, 1):
            key = document_key(doc)

            if key in seen:
                continue

            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)

    return [
        (docs[key], scores[key])
        for key in ranked[:k]
    ]


def retrieve_documents(query, k=TOP_K, document_names=None):
    if not query or not query.strip():
        return []

    store = get_vector_store()

    total = len(store.get(include=[])["ids"])

    if total == 0:
        return []

    # Optionally limit retrieval to documents selected in the UI.
    selected_names = [name for name in (document_names or []) if name]
    where_filter = {"document_name": {"$in": selected_names}} if selected_names else None

    if selected_names:
        filtered = store.get(where=where_filter, include=["metadatas"])
        total = len(filtered.get("ids", []))
        if total == 0:
            return []

    candidate_k = min(CANDIDATE_K, total)

    # Expand the original query into several related queries.
    queries = expand_query(query)

    # Retrieve semantic results for every query.
    semantic_results = []

    for expanded_query in queries:
        search_kwargs = {"k": candidate_k}
        if where_filter:
            search_kwargs["filter"] = where_filter
        results = store.similarity_search(expanded_query, **search_kwargs)
        semantic_results.append(results)

    # Load indexed documents for BM25 lexical retrieval.
    raw = store.get(
        where=where_filter,
        include=["documents", "metadatas"],
    ) if where_filter else store.get(include=["documents", "metadatas"])

    corpus = [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
        for text, metadata in zip(
            raw.get("documents") or [],
            raw.get("metadatas") or [],
        )
        if text and text.strip()
    ]

    lexical_results = []

    if corpus:
        tokenized_corpus = [
            tokens(doc.page_content)
            for doc in corpus
        ]

        bm25 = BM25Okapi(tokenized_corpus)

        # Search BM25 using the original and expanded queries.
        for expanded_query in queries:
            query_tokens = tokens(expanded_query)

            if not query_tokens:
                continue

            scores = bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )[:candidate_k]

            results = [
                corpus[i]
                for i in ranked_indices
                if scores[i] > 0
            ]

            lexical_results.append(results)

    # Fuse all semantic and lexical rankings.
    all_rankings = semantic_results + lexical_results

    return rrf(all_rankings, k=k)
