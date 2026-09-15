import os
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# ============================================================
# 1. Load environment configuration
# ============================================================

load_dotenv()

_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
_CHAT_MODEL: Optional[str] = os.getenv("CHAT_MODEL")
_EMBED_MODEL: Optional[str] = os.getenv("EMBED_MODEL")

# Module-level clients — created lazily on first use
_client: Optional[OpenAI] = None
_chroma_client: Optional[Any] = None
_collection: Optional[Any] = None


def _get_client() -> OpenAI:
    """Return (and lazily create) the OpenAI-compatible client."""
    global _client
    if _client is not None:
        return _client

    if not _API_KEY:
        raise ValueError("OPENAI_API_KEY is missing")
    if not _BASE_URL:
        raise ValueError("OPENAI_BASE_URL is missing")

    _client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    return _client


def _get_collection() -> Any:
    """Return (and lazily create) the ChromaDB collection."""
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    _collection = _chroma_client.get_collection(name="rag_chunks")
    return _collection


# ============================================================
# 4. Create query embedding
# ============================================================

def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string."""
    if not _EMBED_MODEL:
        raise ValueError("EMBED_MODEL is missing")

    client = _get_client()
    response = client.embeddings.create(model=_EMBED_MODEL, input=query)
    return response.data[0].embedding


# ============================================================
# 5. Retrieve relevant chunks
# ============================================================

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks from ChromaDB, converting distance to score."""
    query_vector = embed_query(query)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_chunks = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        # Convert ChromaDB distance to similarity score (0-1 range)
        # Smaller distance = higher similarity
        score = 1.0 / (1.0 + distance)

        retrieved_chunks.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": distance,
            "score": score,  # Add score field for compatibility with guardrails
        })

    return retrieved_chunks


# ============================================================
# 6. Build grounded prompt
# ============================================================

def build_grounded_prompt(
    question: str, retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build a grounded prompt with context and source tracking."""
    context_parts = []
    sources = []

    for chunk in retrieved_chunks:
        source = chunk["metadata"].get("source", "Unknown source")
        section = chunk["metadata"].get("section", "Unknown section")

        context_parts.append(
            f"[Source: {source} | Section: {section}]\n{chunk['text']}"
        )

        sources.append({"id": chunk["id"], "source": source, "section": section})

    context = "\n\n".join(context_parts)

    prompt = f"""
You are an HR assistant.

Answer the user's question using ONLY the information
provided in the retrieved context below.

IMPORTANT RULES:
1. Do not use outside knowledge.
2. Do not invent information.
3. Do not make unsupported claims.
4. If the context does not contain enough information,
   say that there is not enough information in the provided context.
5. Mention the relevant source when possible.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

Provide a concise, accurate answer based only on the context.
"""

    return {"prompt": prompt, "sources": sources}


# ============================================================
# 7. Call Gemini
# ============================================================

def call_llm(prompt: str) -> str:
    """Call the LLM with a grounded prompt."""
    if not _CHAT_MODEL:
        raise ValueError("CHAT_MODEL is missing")

    client = _get_client()
    response = client.chat.completions.create(
        model=_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a grounded HR assistant. Use only the supplied context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ============================================================
# 8. Generate grounded answer
# ============================================================

def generate_grounded_answer(
    question: str, retrieved_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate an answer using only retrieved context."""
    if not retrieved_chunks:
        return {
            "question": question,
            "answer": "I don't have enough information in the provided context.",
            "sources": [],
            "chunks": [],
        }

    prompt_data = build_grounded_prompt(question, retrieved_chunks)
    answer = call_llm(prompt_data["prompt"])

    return {
        "question": question,
        "answer": answer,
        "sources": prompt_data["sources"],
        "chunks": retrieved_chunks,
    }


# ============================================================
# 9. Ungrounded answer
# ============================================================

def generate_ungrounded_answer(question: str) -> str:
    """Generate an answer without retrieval (for comparison only)."""
    prompt = f"Answer this question as an HR assistant:\n\n{question}"
    return call_llm(prompt)


# ============================================================
# 10. Display result
# ============================================================

def display_grounded_result(result: Dict[str, Any]) -> None:
    """Print a formatted grounded answer result."""
    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSupporting Sources:")
    for source in result["sources"]:
        print(f"- {source['id']} | {source['source']} | {source['section']}")

    print("\nSupporting Chunks:")
    for chunk in result["chunks"]:
        print("\nChunk ID:", chunk["id"])
        print("Text:", chunk["text"][:300])


# ============================================================
# 11. Main demonstration
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GROUNDED ANSWER GENERATION")
    print("=" * 70)

    # Task 1 + Task 2
    question = "What are the password reset steps?"
    print("\nQuestion:")
    print(question)

    retrieved_chunks = retrieve(question, k=4)
    print(f"\nRetrieved {len(retrieved_chunks)} chunks.")

    grounded_result = generate_grounded_answer(question, retrieved_chunks)
    display_grounded_result(grounded_result)

    # Task 3 — Missing context
    print("\n" + "=" * 70)
    print("MISSING CONTEXT TEST")
    print("=" * 70)

    missing_context_question = (
        "What is the company's policy for international relocation to Mars?"
    )
    fallback_result = generate_grounded_answer(missing_context_question, [])

    print("\nQuestion:")
    print(missing_context_question)
    print("\nFallback:")
    print(fallback_result["answer"])
    print("\nSources:")
    print(fallback_result["sources"])

    # Task 4 — With vs without retrieval
    print("\n" + "=" * 70)
    print("WITH RETRIEVAL VS WITHOUT RETRIEVAL")
    print("=" * 70)

    comparison_question = "What are the password reset steps?"
    print("\nQuestion:")
    print(comparison_question)

    print("\nWITHOUT RETRIEVAL:")
    ungrounded_answer = generate_ungrounded_answer(comparison_question)
    print(ungrounded_answer)

    print("\nWITH RETRIEVAL:")
    grounded_answer = generate_grounded_answer(
        comparison_question, retrieve(comparison_question, k=4)
    )
    print(grounded_answer["answer"])

    print("\nSources:")
    for source in grounded_answer["sources"]:
        print(f"- {source['source']} ({source['section']})")

    # Final verification
    print("\n" + "=" * 70)
    print("GROUNDING CHECK")
    print("=" * 70)
    print("The grounded answer was generated using retrieved chunks from ChromaDB.")
    print("The answer should not contain claims that are unsupported by those chunks.")
    print("Missing-context questions return a fallback instead of inventing an answer.")
    print("\nEvaluation complete.")
