import json

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import OLLAMA_MODEL, TOP_K
from retriever import retrieve_documents


# --------------------------------------------------
# 1. Structured output schema
# --------------------------------------------------

class BookSummary(BaseModel):
    title: str = Field(
        description="The title of the book, if supported by the excerpts."
    )

    main_topic: str = Field(
        description="The central subject of the book supported by the excerpts."
    )

    key_ideas: list[str] = Field(
        description="Important ideas explicitly supported by the excerpts."
    )

    section_summaries: list[str] = Field(
        description="Summaries of sections represented in the retrieved excerpts."
    )

    practical_takeaways: list[str] = Field(
        description="Practical lessons supported by the retrieved excerpts."
    )

    limitations: list[str] = Field(
        description="Information needed for a complete answer that the excerpts do not establish."
    )


class RAGAnswer(BaseModel):
    answer: str = Field(
        description="The answer to the user's question, based only on retrieved evidence."
    )

    key_points: list[str] = Field(
        description="Important supporting points from the retrieved excerpts."
    )

    sources: list[str] = Field(
        description="Source document and page references supporting the answer."
    )

    context_sufficient: bool = Field(
        description="Whether the retrieved excerpts provide enough evidence to answer."
    )


# --------------------------------------------------
# 2. Prompt templates
# --------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
  
"""
You are a document question-answering assistant.

Answer the user's question using ONLY the supplied excerpts.

STRICT RULES:
1. Treat the supplied document as the source of truth.
2. Prefer the document's exact definitions and terminology over
   your general knowledge.
3. Do not replace the document's wording with a more familiar
   definition from outside the document.
4. Preserve lists, acronyms, names, dates, and technical terms
   exactly as stated in the source.
5. If the document says "realistic and timely", do not substitute
   "relevant and time-bound".
6. Cite the supporting document and page number.
7. Set context_sufficient to true if the excerpts contain enough
   evidence to answer the question; otherwise, set it to false.
8. If the evidence is insufficient, explain what information is missing.
9. Never invent evidence or citations.
10. Return structured output matching the requested schema.

RETRIEVED EXCERPTS:
{context}
"""
    ),
    ("human", "{question}"),
])


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document summarization assistant.

Create a structured summary using ONLY the supplied excerpts.

Rules:
- Do not invent facts or add knowledge from outside the excerpts.
- Identify the title only when supported by the source material.
- Summarize only ideas actually present in the excerpts.
- Do not pretend that a few excerpts represent the entire book.
- If the excerpts cover only a small portion of the book, explain
  this limitation in the limitations field.
- Do not invent chapter names, section summaries, or takeaways.
- Return only a valid JSON object matching the requested schema.
- Use empty lists when the excerpts do not support the information.
- Keep the summary concise and useful.

RETRIEVED EXCERPTS:
{context}
"""
    ),
    ("human", "{question}"),
])


# --------------------------------------------------
# 3. Initialize the local Ollama model
# --------------------------------------------------

model = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0,
)

# Use Pydantic to validate structured model output.
answer_model = model.with_structured_output(
    RAGAnswer,
    method="json_schema",
)

summary_model = model.with_structured_output(
    BookSummary,
    method="json_schema",
)


# --------------------------------------------------
# 4. Format retrieved documents
# --------------------------------------------------

def build_context(results):
    blocks = []

    for i, (doc, score) in enumerate(results, 1):
        metadata = doc.metadata

        document_name = metadata.get("document_name", "Unknown document")
        page_start = metadata.get("page_start", "?")
        page_end = metadata.get("page_end", "?")
        section_path = metadata.get("section_path", "Unknown section")

        blocks.append(
            f"[{i}] {document_name} "
            f"| pages {page_start}-{page_end} "
            f"| section: {section_path} "
            f"| retrieval_score: {score:.4f}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------
# 5. Display structured results
# --------------------------------------------------

def display_json(result):
    if hasattr(result, "model_dump"):
        data = result.model_dump()
    else:
        data = result

    print("\n--- STRUCTURED OUTPUT (JSON) ---\n")
    print(json.dumps(data, indent=2, ensure_ascii=False))


# --------------------------------------------------
# 6. Answer normal questions
# --------------------------------------------------

def answer_question(question, context):
    messages = ANSWER_PROMPT.format_messages(
        question=question,
        context=context,
    )

    result = answer_model.invoke(messages)
    display_json(result)


# --------------------------------------------------
# 7. Summarize retrieved content
# --------------------------------------------------

def summarize_book(question, context):
    messages = SUMMARY_PROMPT.format_messages(
        question=question,
        context=context,
    )

    result = summary_model.invoke(messages)
    display_json(result)


# --------------------------------------------------
# 8. Main RAG function
# --------------------------------------------------

def ask(question):
    print("\nRetrieving relevant passages...")

    results = retrieve_documents(question, TOP_K)

    if not results:
        print(
            "No passages found. Add a PDF to the documents/ folder "
            "and run: uv run python ingest.py"
        )
        return

    context = build_context(results)

    print("\n--- RETRIEVED EXCERPTS ---\n")
    print(context)

    question_lower = question.lower()

    summary_keywords = (
        "summarize",
        "summary",
        "summarise",
        "overview of this book",
        "what is this book about",
    )

    is_summary_question = any(
        keyword in question_lower
        for keyword in summary_keywords
    )

    print("\n--- ANSWER ---")

    if is_summary_question:
        summarize_book(question, context)
    else:
        answer_question(question, context)


# --------------------------------------------------
# 9. Interactive terminal
# --------------------------------------------------

def main():
    print(f"General RAG using Ollama model: {OLLAMA_MODEL}")
    print("Structured output: Pydantic + JSON")
    print("Type exit to quit.")

    while True:
        try:
            question = input("\nQuestion: ").strip()

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if question:
                ask(question)

        except KeyboardInterrupt:
            print("\nGoodbye.")
            break

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
