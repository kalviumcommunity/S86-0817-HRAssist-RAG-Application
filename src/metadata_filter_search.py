import os
from dotenv import load_dotenv
from openai import OpenAI

import chromadb


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not BASE_URL:
    raise ValueError("OPENAI_BASE_URL is missing from .env")

if not EMBED_MODEL:
    raise ValueError("EMBED_MODEL is missing from .env")


# --------------------------------------------------
# 2. Create Gemini OpenAI-compatible client
# --------------------------------------------------

openai_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# --------------------------------------------------
# 3. Connect to existing ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

COLLECTION_NAME = "rag_chunks"

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# 4. Create query embedding
# --------------------------------------------------

def embed_query(query):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )

    return response.data[0].embedding


# --------------------------------------------------
# 5. Vector retrieval with optional metadata filter
# --------------------------------------------------

def retrieve(query, k=3, metadata_filter=None):

    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=metadata_filter,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    formatted_results = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        # Chroma returns distance.
        # Smaller distance = more similar.
        similarity_score = 1 / (1 + distance)

        formatted_results.append({
            "score": similarity_score,
            "text": document,
            "metadata": metadata
        })

    return formatted_results


# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

def show_results(label, results):

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for index, item in enumerate(results, start=1):

        metadata = item["metadata"]

        print(f"\nResult {index}")
        print("-" * 40)

        print("score:", round(item["score"], 4))

        print(
            "source:",
            metadata.get("source", "N/A")
        )

        print(
            "section:",
            metadata.get("section", "N/A")
        )

        print(
            "document_type:",
            metadata.get("document_type", "N/A")
        )

        print(
            "user_group:",
            metadata.get("user_group", "N/A")
        )

        print(
            "text:",
            item["text"][:250]
        )


# --------------------------------------------------
# 7. Keyword scoring
# --------------------------------------------------

def keyword_score(text, keywords):

    lowered_text = text.lower()

    return sum(
        1
        for keyword in keywords
        if keyword.lower() in lowered_text
    )


# --------------------------------------------------
# 8. Hybrid ranking
# --------------------------------------------------

def hybrid_rank(
    vector_results,
    keywords,
    vector_weight=0.8,
    keyword_weight=0.2
):

    ranked = []

    for item in vector_results:

        lexical_score = keyword_score(
            item["text"],
            keywords
        )

        # Normalize keyword score
        max_keyword_score = max(len(keywords), 1)

        normalized_keyword_score = (
            lexical_score / max_keyword_score
        )

        combined_score = (
            vector_weight * item["score"]
            + keyword_weight * normalized_keyword_score
        )

        ranked.append({
            **item,
            "keyword_score": lexical_score,
            "hybrid_score": combined_score
        })

    return sorted(
        ranked,
        key=lambda item: item["hybrid_score"],
        reverse=True
    )


# --------------------------------------------------
# 9. Main demonstration
# --------------------------------------------------

if __name__ == "__main__":

    query = "What are the password reset steps?"

    metadata_filter = {
        "section": "Account access"
    }

    print("\nMETADATA FILTERING AND HYBRID SEARCH DEMO")

    print("\nQuery:")
    print(query)

    print("\nFilter:")
    print(metadata_filter)


    # ----------------------------------------------
    # Task 1 + Task 2
    # Unfiltered search
    # ----------------------------------------------

    unfiltered = retrieve(
        query,
        k=3
    )

    show_results(
        "UNFILTERED VECTOR SEARCH",
        unfiltered
    )


    # ----------------------------------------------
    # Filtered search
    # ----------------------------------------------

    filtered = retrieve(
        query,
        k=3,
        metadata_filter=metadata_filter
    )

    show_results(
        "FILTERED VECTOR SEARCH",
        filtered
    )


    # ----------------------------------------------
    # Task 3
    # Hybrid search
    # ----------------------------------------------

    keywords = [
        "password",
        "reset"
    ]

    hybrid = hybrid_rank(
        filtered,
        keywords=keywords
    )


    print()
    print("=" * 70)
    print("HYBRID SEARCH")
    print("=" * 70)

    print("Keywords:", keywords)

    for index, item in enumerate(hybrid, start=1):

        metadata = item["metadata"]

        print(f"\nResult {index}")
        print("-" * 40)

        print(
            "vector_score:",
            round(item["score"], 4)
        )

        print(
            "keyword_score:",
            item["keyword_score"]
        )

        print(
            "hybrid_score:",
            round(item["hybrid_score"], 4)
        )

        print(
            "source:",
            metadata.get("source", "N/A")
        )

        print(
            "section:",
            metadata.get("section", "N/A")
        )

        print(
            "text:",
            item["text"][:250]
        )


    # ----------------------------------------------
    # Task 4
    # Precision demonstration
    # ----------------------------------------------

    print()
    print("=" * 70)
    print("PRECISION COMPARISON")
    print("=" * 70)

    print(
        "\nUnfiltered results search the entire vector database."
    )

    print(
        "Filtered results are restricted to the "
        "'Account access' section."
    )

    print(
        "\nThis improves precision when the metadata correctly "
        "represents the user's intent."
    )

    print(
        "Hybrid ranking further rewards exact keyword matches "
        "such as 'password' and 'reset'."
    )