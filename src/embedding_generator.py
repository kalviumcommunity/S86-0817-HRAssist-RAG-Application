import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIError,
)

from src.similarity import cosine_similarity


# Load environment variables from .env
load_dotenv()

# Read configuration from .env — kept as module-level vars but validation
# is deferred to _get_client() so this module can be safely imported without
# a configured .env (e.g. in tests or when used as a library).
_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
_EMBED_MODEL: Optional[str] = os.getenv("EMBED_MODEL")

# Public alias used by callers that just need the model name string
EMBED_MODEL: str = _EMBED_MODEL or ""

# Module-level client — created lazily on first use
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return (and lazily create) the OpenAI-compatible client.

    Raises ``ValueError`` with a clear message if required env vars are
    absent.  Deferring this check to call-time means the module can always
    be imported safely.
    """
    global _client
    if _client is not None:
        return _client

    if not _API_KEY:
        raise ValueError("OPENAI_API_KEY is missing from the .env file")
    if not _EMBED_MODEL:
        raise ValueError("EMBED_MODEL is missing from the .env file")

    _client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
    return _client


# Small HR-related corpus used for chunk-level embedding demos
chunks = [
    {
        "text": (
            "Employees are entitled to annual leave based on "
            "their employment status and company policy."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 0,
            "section": "Annual Leave",
        },
    },
    {
        "text": (
            "Employees must submit leave requests through the "
            "HR portal at least five working days in advance."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 1,
            "section": "Leave Request Process",
        },
    },
    {
        "text": (
            "Sick leave is available when an employee is unable "
            "to work because of illness or a medical condition."
        ),
        "metadata": {
            "source": "employee_leave_policy.txt",
            "chunk_index": 2,
            "section": "Sick Leave",
        },
    },
]


def embed(texts: List[str]) -> List[List[float]]:
    """Generate embedding vectors for a list of raw text strings.

    Sends all texts in a single API request and returns a list of float
    vectors in the same order as the input. Each vector's length (dimension)
    is determined by the embedding model.

    Args:
        texts: Plain text strings to embed.

    Returns:
        A list of float vectors, one per input text.
    """
    client = _get_client()
    response = client.embeddings.create(
        model=_EMBED_MODEL,  # type: ignore[arg-type]  — validated in _get_client
        input=texts,
    )
    # response.data is ordered to match the input list
    return [item.embedding for item in response.data]


def generate_embeddings(chunk_list: List[dict]) -> List[dict]:
    """Generate embeddings for prepared text chunks.

    Each embedding is stored with its original text and metadata.

    Args:
        chunk_list: List of dicts with 'text' and 'metadata' keys.

    Returns:
        List of records combining text, metadata, and embedding vector.
    """
    texts = [chunk["text"] for chunk in chunk_list]

    print(f"\nGenerating embeddings using: {_EMBED_MODEL}")
    print(f"Number of chunks: {len(texts)}")

    embeddings = embed(texts)

    records = []
    for chunk, embedding in zip(chunk_list, embeddings):
        record = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": embedding,
        }
        records.append(record)

    return records


def demonstrate_vector_dimension(embeddings: List[List[float]]) -> None:
    """Report the vector dimension and a sample of the first embedding.

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


def demonstrate_semantic_similarity(
    embeddings: List[List[float]], texts: List[str]
) -> None:
    """Compare a semantically similar pair against a dissimilar pair.

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


def print_results(records: List[dict]) -> None:
    """Print a formatted summary of all embedding records."""

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION RESULTS")
    print("=" * 70)

    print(f"\nEmbedding model: {_EMBED_MODEL}")
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


def main() -> None:
    """Run the embedding generation demo."""

    print("=" * 70)
    print("GENERATING EMBEDDINGS VIA GEMINI API")
    print("=" * 70)

    # ── Part 1: sample texts demonstrating semantic meaning ──────────────
    sample_texts = [
        "How do I reset my account password?",
        "Steps to recover access to my login",
        "The cafeteria menu has pasta today",
    ]

    try:

        print("\n" + "=" * 70)
        print("PART 1 — VECTOR DIMENSION & SEMANTIC SIMILARITY DEMO")
        print("=" * 70)

        print(f"\nEmbedding {len(sample_texts)} sample texts with {_EMBED_MODEL} ...")
        sample_embeddings = embed(sample_texts)

        demonstrate_vector_dimension(sample_embeddings)
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
