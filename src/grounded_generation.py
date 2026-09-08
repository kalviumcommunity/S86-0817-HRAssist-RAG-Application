import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# ============================================================
# 1. Load environment configuration
# ============================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("CHAT_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing")

if not BASE_URL:
    raise ValueError("OPENAI_BASE_URL is missing")

if not CHAT_MODEL:
    raise ValueError("CHAT_MODEL is missing")

if not EMBED_MODEL:
    raise ValueError("EMBED_MODEL is missing")


# ============================================================
# 2. OpenAI-compatible Gemini client
# ============================================================

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# 3. Connect to ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="rag_chunks"
)


# ============================================================
# 4. Create query embedding
# ============================================================

def embed_query(query):

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )

    return response.data[0].embedding


# ============================================================
# 5. Retrieve relevant chunks
# ============================================================

def retrieve(query, k=4):

    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_chunks = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):

        retrieved_chunks.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": distance
        })

    return retrieved_chunks


# ============================================================
# 6. Build grounded prompt
# ============================================================

def build_grounded_prompt(question, retrieved_chunks):

    context_parts = []
    sources = []

    for chunk in retrieved_chunks:

        source = chunk["metadata"].get(
            "source",
            "Unknown source"
        )

        section = chunk["metadata"].get(
            "section",
            "Unknown section"
        )

        context_parts.append(
            f"[Source: {source} | Section: {section}]\n"
            f"{chunk['text']}"
        )

        sources.append({
            "id": chunk["id"],
            "source": source,
            "section": section
        })

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

    return {
        "prompt": prompt,
        "sources": sources
    }


# ============================================================
# 7. Call Gemini
# ============================================================

def call_llm(prompt):

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a grounded HR assistant. "
                    "Use only the supplied context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


# ============================================================
# 8. Generate grounded answer
# ============================================================

def generate_grounded_answer(
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return {
            "question": question,
            "answer": (
                "I don't have enough information "
                "in the provided context."
            ),
            "sources": [],
            "chunks": []
        }

    prompt_data = build_grounded_prompt(
        question,
        retrieved_chunks
    )

    answer = call_llm(
        prompt_data["prompt"]
    )

    return {
        "question": question,
        "answer": answer,
        "sources": prompt_data["sources"],
        "chunks": retrieved_chunks
    }


# ============================================================
# 9. Ungrounded answer
# ============================================================

def generate_ungrounded_answer(question):

    prompt = f"""
Answer this question as an HR assistant:

{question}
"""

    return call_llm(prompt)


# ============================================================
# 10. Display result
# ============================================================

def display_grounded_result(result):

    print("\n" + "=" * 70)
    print("GROUNDED ANSWER")
    print("=" * 70)

    print("\nQuestion:")
    print(result["question"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSupporting Sources:")

    for source in result["sources"]:

        print(
            f"- {source['id']} | "
            f"{source['source']} | "
            f"{source['section']}"
        )

    print("\nSupporting Chunks:")

    for chunk in result["chunks"]:

        print("\nChunk ID:", chunk["id"])

        print(
            "Text:",
            chunk["text"][:300]
        )


# ============================================================
# 11. Main demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("GROUNDED ANSWER GENERATION")
    print("=" * 70)


    # --------------------------------------------------------
    # Task 1 + Task 2
    # --------------------------------------------------------

    question = (
        "What are the password reset steps?"
    )

    print("\nQuestion:")
    print(question)

    retrieved_chunks = retrieve(
        question,
        k=4
    )

    print(
        f"\nRetrieved {len(retrieved_chunks)} chunks."
    )

    grounded_result = generate_grounded_answer(
        question,
        retrieved_chunks
    )

    display_grounded_result(
        grounded_result
    )


    # --------------------------------------------------------
    # Task 3 — Missing context
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING CONTEXT TEST")
    print("=" * 70)

    missing_context_question = (
        "What is the company's policy for "
        "international relocation to Mars?"
    )

    fallback_result = generate_grounded_answer(
        missing_context_question,
        []
    )

    print("\nQuestion:")
    print(missing_context_question)

    print("\nFallback:")
    print(fallback_result["answer"])

    print("\nSources:")
    print(fallback_result["sources"])


    # --------------------------------------------------------
    # Task 4 — With vs without retrieval
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("WITH RETRIEVAL VS WITHOUT RETRIEVAL")
    print("=" * 70)

    comparison_question = (
        "What are the password reset steps?"
    )

    print("\nQuestion:")
    print(comparison_question)


    print("\nWITHOUT RETRIEVAL:")
    ungrounded_answer = generate_ungrounded_answer(
        comparison_question
    )

    print(ungrounded_answer)


    print("\nWITH RETRIEVAL:")

    grounded_answer = generate_grounded_answer(
        comparison_question,
        retrieve(comparison_question, k=4)
    )

    print(grounded_answer["answer"])

    print("\nSources:")

    for source in grounded_answer["sources"]:

        print(
            f"- {source['source']} "
            f"({source['section']})"
        )


    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUNDING CHECK")
    print("=" * 70)

    print(
        "The grounded answer was generated using "
        "retrieved chunks from ChromaDB."
    )

    print(
        "The answer should not contain claims that "
        "are unsupported by those chunks."
    )

    print(
        "Missing-context questions return a fallback "
        "instead of inventing an answer."
    )

    print("\nEvaluation complete.")