import os
import chromadb

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIError
)


# -----------------------------------------
# Load environment configuration
# -----------------------------------------

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")


# -----------------------------------------
# Vector dimension
# -----------------------------------------
# IMPORTANT:
# Replace this with the actual vector
# length returned by your embedding API.

VECTOR_DIMENSION = 3072


# -----------------------------------------
# Collection configuration
# -----------------------------------------

COLLECTION_NAME = "rag_chunks"


# -----------------------------------------
# Create Gemini OpenAI-compatible client
# -----------------------------------------

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# -----------------------------------------
# Create persistent ChromaDB client
# -----------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)


# -----------------------------------------
# Create / retrieve collection
# -----------------------------------------

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "description": "HR Assist RAG document chunks"
    },
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    }
)


def main():

    print("=" * 70)
    print("VECTOR DATABASE SETUP & COLLECTION TEST")
    print("=" * 70)

    print("\nEmbedding model:", EMBED_MODEL)
    print("Collection:", COLLECTION_NAME)
    print("Expected vector dimension:", VECTOR_DIMENSION)

    # -----------------------------------------
    # Test document chunk
    # -----------------------------------------

    test_text = (
        "Employees must submit leave requests through "
        "the HR portal at least five working days in advance."
    )

    test_metadata = {
        "source": "employee_leave_policy.txt",
        "chunk_index": 0,
        "section": "Leave Request Process"
    }

    record_id = "employee_leave_policy.txt:0"

    try:

        # -----------------------------------------
        # Generate embedding
        # -----------------------------------------

        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=[test_text]
        )

        embedding = response.data[0].embedding

        print(
            "\nGenerated vector length:",
            len(embedding)
        )

        # -----------------------------------------
        # Verify vector dimension
        # -----------------------------------------

        if len(embedding) != VECTOR_DIMENSION:

            raise ValueError(
                f"Vector dimension mismatch. "
                f"Expected {VECTOR_DIMENSION}, "
                f"got {len(embedding)}"
            )

        print("Vector dimension check: PASSED")

        # -----------------------------------------
        # Insert record
        # -----------------------------------------

        collection.upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[test_text],
            metadatas=[test_metadata]
        )

        print("\nRecord inserted successfully.")

        # -----------------------------------------
        # Read record back
        # -----------------------------------------

        stored = collection.get(
            ids=[record_id],
            include=[
                "embeddings",
                "documents",
                "metadatas"
            ]
        )

        print("\n" + "=" * 70)
        print("READBACK RESULT")
        print("=" * 70)

        print("\nID:")
        print(stored["ids"][0])

        print("\nVector length:")
        print(len(stored["embeddings"][0]))

        print("\nText:")
        print(stored["documents"][0])

        print("\nMetadata:")
        print(stored["metadatas"][0])

        print("\nVector sample:")
        print(stored["embeddings"][0][:5])

        print("\n" + "=" * 70)
        print("VECTOR DATABASE TEST PASSED")
        print("=" * 70)

    except AuthenticationError:

        print(
            "\nERROR: Authentication failed (401)."
        )

        print(
            "Check OPENAI_API_KEY in your .env file."
        )

    except RateLimitError:

        print(
            "\nERROR: Rate limit reached (429)."
        )

        print(
            "Please wait and try again."
        )

    except APIError as error:

        print("\nAPI ERROR:")
        print(error)

    except Exception as error:

        print("\nERROR:")
        print(error)


if __name__ == "__main__":
    main()