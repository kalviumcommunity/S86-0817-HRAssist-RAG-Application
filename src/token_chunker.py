import tiktoken


encoding = tiktoken.get_encoding("cl100k_base")


def token_chunks(text, chunk_size=100, overlap=20):
    """
    Split text into chunks based on token count.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    tokens = encoding.encode(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = start + chunk_size

        chunk_tokens = tokens[start:end]

        chunk_text = encoding.decode(chunk_tokens)

        chunks.append(chunk_text)

        # Move forward while preserving overlap
        start += chunk_size - overlap

    return chunks


def show_chunks(chunks):

    print("\n" + "=" * 70)
    print("TOKEN-AWARE CHUNKING RESULTS")
    print("=" * 70)

    print(f"\nTotal chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):

        token_count = len(encoding.encode(chunk))

        print("\n" + "-" * 70)
        print(f"CHUNK {index}")
        print(f"Token count: {token_count}")
        print("-" * 70)

        print(chunk)


def demonstrate_overlap(text):

    print("\n" + "=" * 70)
    print("OVERLAP DEMONSTRATION")
    print("=" * 70)

    chunk_size = 40

    no_overlap_chunks = token_chunks(
        text,
        chunk_size=chunk_size,
        overlap=0
    )

    overlap_chunks = token_chunks(
        text,
        chunk_size=chunk_size,
        overlap=10
    )

    print("\nWITHOUT OVERLAP")
    print(f"Chunks: {len(no_overlap_chunks)}")

    for i, chunk in enumerate(no_overlap_chunks):

        print(f"\nChunk {i + 1}:")
        print(chunk)

    print("\n" + "=" * 70)

    print("\nWITH 10 TOKEN OVERLAP")
    print(f"Chunks: {len(overlap_chunks)}")

    for i, chunk in enumerate(overlap_chunks):

        print(f"\nChunk {i + 1}:")
        print(chunk)


def verify_overlap(text, chunk_size, overlap):

    chunks = token_chunks(
        text,
        chunk_size,
        overlap
    )

    print("\n" + "=" * 70)
    print("OVERLAP VERIFICATION")
    print("=" * 70)

    for i in range(len(chunks) - 1):

        current_tokens = encoding.encode(chunks[i])
        next_tokens = encoding.encode(chunks[i + 1])

        expected_overlap = current_tokens[-overlap:]
        actual_overlap = next_tokens[:overlap]

        matches = expected_overlap == actual_overlap

        print(f"\nChunk {i + 1} → Chunk {i + 2}")
        print(f"Expected overlap: {overlap} tokens")
        print(f"Overlap preserved: {matches}")


def explain_configuration():

    print("\n" + "=" * 70)
    print("CHUNK SIZE & OVERLAP JUSTIFICATION")
    print("=" * 70)

    print("""
Chunk size: 100 tokens
Overlap: 20 tokens

Why 100 tokens?

A 100-token chunk provides focused retrieval while retaining
enough context for HR policy information.

Why 20-token overlap?

20 tokens represent 20% overlap. This helps preserve context
when important information occurs near chunk boundaries.

Trade-off:

More overlap preserves more context but increases duplicate
tokens, embedding cost, and storage.

This configuration balances retrieval quality, context
preservation, and computational cost.
""")


sample_text = """
HR Assist Employee Leave Policy

Employees are entitled to different types of leave depending on
their employment status and company policy.

Annual leave allows employees to take planned time away from work.
Employees must submit annual leave requests through the HR portal
at least five working days before the requested leave period.

Sick leave is available when an employee is unable to work because
of illness or a medical condition. Employees may be required to
provide medical documentation for extended periods of absence.

The HR manager reviews leave requests and approves or rejects them
based on staffing requirements and company policy.

Employees should contact the HR department if they need assistance
understanding their available leave balance or submitting a request.
"""


def main():

    print("=" * 70)
    print("TOKEN-AWARE CHUNK SIZING & OVERLAP")
    print("=" * 70)

    chunk_size = 100
    overlap = 20

    chunks = token_chunks(
        sample_text,
        chunk_size,
        overlap
    )

    show_chunks(chunks)

    demonstrate_overlap(sample_text)

    verify_overlap(
        sample_text,
        chunk_size=40,
        overlap=10
    )

    explain_configuration()


if __name__ == "__main__":
    main()