import os

from dotenv import load_dotenv
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIError
)


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


# Small HR-related corpus
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


def generate_embeddings(chunks):
    """
    Generate embeddings for prepared text chunks.
    Each embedding is stored with its original
    text and metadata.
    """

    # Extract text from all chunks
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(f"\nGenerating embeddings using: {EMBED_MODEL}")
    print(f"Number of chunks: {len(texts)}")

    # Send all chunks in one API request
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    records = []

    # Combine text + metadata + embedding
    for chunk, item in zip(chunks, response.data):

        record = {
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "embedding": item.embedding
        }

        records.append(record)

    return records


def print_results(records):

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION RESULTS")
    print("=" * 70)

    print(f"\nEmbedding model: {EMBED_MODEL}")

    print(
        f"Number of chunks embedded: {len(records)}"
    )

    if records:

        first_record = records[0]

        print(
            f"Vector length: "
            f"{len(first_record['embedding'])}"
        )

        print(
            "Sample vector values: "
            f"{first_record['embedding'][:5]}"
        )

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

        print(
            "\nVector length:",
            len(record["embedding"])
        )

        print(
            "Vector sample:",
            record["embedding"][:5]
        )


def main():

    print("=" * 70)
    print("GENERATING EMBEDDINGS VIA GEMINI API")
    print("=" * 70)

    try:

        records = generate_embeddings(chunks)

        print_results(records)

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

        print("\nUNEXPECTED ERROR:")
        print(error)


if __name__ == "__main__":
    main()