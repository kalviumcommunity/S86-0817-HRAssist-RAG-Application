import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb


# ==================================================
# 1. Load environment variables
# ==================================================

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


# ==================================================
# 2. Gemini OpenAI-compatible client
# ==================================================

openai_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ==================================================
# 3. Connect to existing ChromaDB
# ==================================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="rag_chunks"
)


# ==================================================
# 4. Generate query embedding
# ==================================================

def embed_query(query):

    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    )

    return response.data[0].embedding


# ==================================================
# 5. Retrieve top-k chunks
# ==================================================

def retrieve(query, k=5):

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

    retrieved = []

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

        similarity_score = 1 / (1 + distance)

        retrieved.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "score": similarity_score
        })

    return retrieved


# ==================================================
# 6. Labelled query set
# ==================================================
#
# IMPORTANT:
# Replace the chunk IDs below with IDs from
# your actual ChromaDB collection.
#

labelled_queries = [
    {
        "query": "What are the password reset steps?",
        "relevant_chunk_ids": {
            "REPLACE_WITH_PASSWORD_CHUNK_ID"
        }
    },

    {
        "query": "What are the employee attendance rules?",
        "relevant_chunk_ids": {
            "REPLACE_WITH_ATTENDANCE_CHUNK_ID"
        }
    },

    {
        "query": "What evidence is required for project submission?",
        "relevant_chunk_ids": {
            "REPLACE_WITH_SUBMISSION_CHUNK_ID"
        }
    }
]


# ==================================================
# 7. Evaluate one query
# ==================================================

def evaluate_query(item, k=5):

    results = retrieve(
        item["query"],
        k=k
    )

    retrieved_ids = [
        result["id"]
        for result in results
    ]

    relevant = item["relevant_chunk_ids"]

    hits = [
        chunk_id
        for chunk_id in retrieved_ids
        if chunk_id in relevant
    ]

    recall = (
        len(hits) / len(relevant)
        if relevant
        else 0
    )

    precision = (
        len(hits) / len(retrieved_ids)
        if retrieved_ids
        else 0
    )

    return {
        "query": item["query"],
        "retrieved_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant),
        "hits": hits,
        "recall": recall,
        "precision": precision,
        "results": results
    }


# ==================================================
# 8. Main evaluation
# ==================================================

if __name__ == "__main__":

    print("=" * 70)
    print("RETRIEVAL EVALUATION & RECALL TESTING")
    print("=" * 70)

    k = 5

    rows = []

    for item in labelled_queries:

        # Skip placeholder IDs
        if any(
            chunk_id.startswith("REPLACE_WITH")
            for chunk_id in item["relevant_chunk_ids"]
        ):
            print("\nSkipping query because its chunk ID has")
            print("not been replaced yet:")
            print(item["query"])
            continue

        result = evaluate_query(
            item,
            k=k
        )

        rows.append(result)

        print("\n" + "-" * 70)
        print("QUERY:")
        print(result["query"])

        print("\nEXPECTED CHUNKS:")
        print(result["relevant_chunk_ids"])

        print("\nRETRIEVED CHUNKS:")
        print(result["retrieved_ids"])

        print("\nHITS:")
        print(result["hits"])

        print(
            "\nRecall@5:",
            round(result["recall"], 3)
        )

        print(
            "Precision@5:",
            round(result["precision"], 3)
        )

        print("\nRetrieved results:")

        for index, retrieved in enumerate(
            result["results"],
            start=1
        ):

            print(f"\nResult {index}")
            print("ID:", retrieved["id"])
            print(
                "Score:",
                round(retrieved["score"], 4)
            )
            print(
                "Metadata:",
                retrieved["metadata"]
            )
            print(
                "Text:",
                retrieved["text"][:200]
            )


    # ==================================================
    # 9. Aggregate metrics
    # ==================================================

    if rows:

        avg_recall = (
            sum(row["recall"] for row in rows)
            / len(rows)
        )

        avg_precision = (
            sum(row["precision"] for row in rows)
            / len(rows)
        )

        print("\n")
        print("=" * 70)
        print("OVERALL RESULTS")
        print("=" * 70)

        print("Queries evaluated:", len(rows))

        print(
            "Average Recall@5:",
            round(avg_recall, 3)
        )

        print(
            "Average Precision@5:",
            round(avg_precision, 3)
        )


        # ==================================================
        # 10. Failure analysis
        # ==================================================

        failures = [
            row
            for row in rows
            if row["recall"] < 1.0
        ]

        print("\n")
        print("=" * 70)
        print("FAILURE ANALYSIS")
        print("=" * 70)

        if not failures:

            print(
                "No recall failures detected."
            )

        else:

            for failure in failures:

                print("\nFailed query:")
                print(failure["query"])

                print(
                    "Expected:",
                    failure["relevant_chunk_ids"]
                )

                print(
                    "Retrieved:",
                    failure["retrieved_ids"]
                )

                print(
                    "Possible causes:"
                )

                print(
                    "- Chunking may separate relevant information."
                )

                print(
                    "- Query wording may differ from the source."
                )

                print(
                    "- Top-k may be too small."
                )

                print(
                    "- Embedding quality may be insufficient."
                )

                print(
                    "- Metadata filtering may be missing."
                )

        print("\nEvaluation complete.")