import os
from typing import List

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIError
)

from src.similarity import cosine_similarity


# Load environment variables from .env
load_dotenv()


# Read configuration from .env
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
EMBED_MODEL = os.getenv("EMBED_MODEL")


# Check required configuration
if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is missing from the .env file"
    )

if not EMBED_MODEL:
    raise ValueError(
        "EMBED_MODEL is missing from the .env file"
    )


# Create OpenAI-compatible client for Gemini API
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# Small HR-related corpus used for chunk-level embedding
chunks = [
    {
        "text": (
            "Employees are entitled to annual leave based on "
            "their employment status and company policy."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 0,
            "section": "Annual Leave"
        }
    },
    {
        "text": (
            "Employees must submit leave requests through the "
            "HR portal at least five working days in advance."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 1,
            "section": "Leave Request Process"
        }
    },
    {
        "text": (
            "Sick leave is available when an employee is unable "
            "to work because of illness or a medical condition."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 2,
            "section": "Sick Leave"
        }
    }
]


def embed(texts: List[str]) -> List[List[float]]:
    """
    Generate embedding vectors for a list of raw text strings.

    Sends all texts in a single API request and returns a list of
    float vectors in the same order as the input. Each vector's
    length (dimension) is determined by the embedding model.

    Args:
        texts: Plain text strings to embed.

    Returns:
        A list of float vectors, one per input text.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    # response.data is ordered to match the input list
    return [item.embedding for item in response.data]


def generate_embeddings(chunks):
    """
    Generate embeddings for prepared text chunks.
    Each embedding is stored with its original text and metadata.

    Args:
        chunks: List of dicts with 'text' and 'metadata' keys.

    Returns:
        List of records combining text, metadata, and embedding vector.
    """

    texts = [chunk["text"] for chunk in chunks]

    print(f"\nGenerating embeddings using: {EMBED_MODEL}")
    print(f"Number of chunks: {len(texts)}")

    embeddings = embed(texts)

    records = []
    for chunk, embedding in zip(chunks, embeddings):
        record = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding
        }
        records.append(record)

    return records


def demonstrate_vector_dimension(embeddings: List[List[float]]) -> None:
    """
    Report the vector dimension and a sample of the first embedding.

    The dimension is the number of numeric coordinates in each vector.
    For example, gemini-embedding-001 produces 3072-dimensional vectors.
    Every text, regardless of length, maps to the same fixed-size vector.

    Args:
        embeddings: List of embedding vectors returned by embed().
    """
    dimension = len(embeddings[0])
    print(f"dimension: {dimension}")
    print(f"first 8 values: {embeddings[0][:8]}")
    print(
        "\nWhat this means: every text is represented by "
        f"{dimension} numeric coordinates. The full pattern "
        "across all dimensions captures the semantic meaning — "
        "no single coordinate has a human-readable interpretation."
    )


def demonstrate_semantic_similarity(embeddings: List[List[float]], texts: List[str]) -> None:
    """
    Compare a semantically similar pair against a dissimilar pair using
    cosine similarity to validate that the embedding model captures meaning.

    Cosine similarity measures the angle between two vectors in high-dimensional
    space. A score near 1.0 means the vectors point in the same direction
    (similar meaning); a score near 0.0 or below means unrelated or opposite.

    Args:
        embeddings: Embedding vectors aligned with ``texts``.
        texts: The original text strings, used for display labels.
    """
    similar_score = cosine_similarity(embeddings[0], embeddings[1])
    dissimilar_score = cosine_similarity(embeddings[0], embeddings[2])

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY COMPARISON")
    print("=" * 70)

    print(f'\nText A: "{texts[0]}"')
    print(f'Text B: "{texts[1]}"')
    print(f'Text C: "{texts[2]}"')

    print(f"\nA vs B (similar meaning):    {similar_score:.6f}")
    print(f"A vs C (dissimilar meaning): {dissimilar_score:.6f}")

    if similar_score > dissimilar_score:
        print(
            "\nResult: PASSED — the similar pair scored higher than the "
            "dissimilar pair. The embedding model correctly captures meaning."
        )
    else:
        print(
            "\nResult: UNEXPECTED — the dissimilar pair scored higher. "
            "Check that the embedding model is loaded correctly."
        )

    print(
        "\nWhy cosine similarity works here: it compares the direction of "
        "two vectors, not their magnitude. Texts about the same topic point "
        "in a similar direction in vector space even when the exact words differ."
    )


def print_results(records):
    """Print a formatted summary of all embedding records."""

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION RESULTS")
    print("=" * 70)

    print(f"\nEmbedding model: {EMBED_MODEL}")
    print(f"Number of chunks embedded: {len(records)}")

    if records:
        first_record = records[0]
        print(f"Vector length: {len(first_record['embedding'])}")
        print(f"Sample vector values: {first_record['embedding'][:5]}")

    print("\n" + "=" * 70)
    print("STORED EMBEDDING RECORDS")
    print("=" * 70)

    for index, record in enumerate(records, start=1):

        print(f"\nRECORD {index}")
        print("-" * 50)

        print("\nText:")
        print(record["text"])

        print("\nMetadata:")
        print(record["metadata"])

        print("\nVector length:", len(record["embedding"]))
        print("Vector sample:", record["embedding"][:5])


def main():

    print("=" * 70)
    print("GENERATING EMBEDDINGS VIA GEMINI API")
    print("=" * 70)

    # ── Part 1: sample texts demonstrating semantic meaning ──────────────
    # These three sentences are chosen deliberately:
    #   [0] and [1] share meaning (password reset ≈ account recovery)
    #   [0] and [2] are unrelated (password reset vs cafeteria menu)
    sample_texts = [
        "How do I reset my account password?",
        "Steps to recover access to my login",
        "The cafeteria menu has pasta today",
    ]

    try:

        print("\n" + "=" * 70)
        print("PART 1 — VECTOR DIMENSION & SEMANTIC SIMILARITY DEMO")
        print("=" * 70)

        print(f"\nEmbedding {len(sample_texts)} sample texts with {EMBED_MODEL} ...")
        sample_embeddings = embed(sample_texts)

        # Report dimension and first 8 values
        demonstrate_vector_dimension(sample_embeddings)

        # Compare similar vs dissimilar pairs
        demonstrate_semantic_similarity(sample_embeddings, sample_texts)

        # ── Part 2: chunk-level embedding for the HR corpus ──────────────
        print("\n" + "=" * 70)
        print("PART 2 — HR CORPUS CHUNK EMBEDDINGS")
        print("=" * 70)

        records = generate_embeddings(chunks)
        print_results(records)

    except AuthenticationError:
        print("\nERROR: Authentication failed (401).")
        print("Check OPENAI_API_KEY in your .env file.")

    except RateLimitError:
        print("\nERROR: Rate limit reached (429).")
        print("Please wait and try again.")

    except APIError as error:
        print("\nAPI ERROR:")
        print(error)

    except Exception as error:
        print("\nUNEXPECTED ERROR:")
        print(error)


if __name__ == "__main__":
    main()