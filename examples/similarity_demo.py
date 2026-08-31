"""Demonstrate cosine similarity ranking for retrieved chunk records.

This demo shows two things:
1. How cosine similarity ranks chunks against a query using synthetic vectors.
2. How the same logic applies to semantic meaning: similar texts score higher
   than unrelated texts, which is the foundation of embedding-based retrieval.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.similarity import cosine_similarity, compare_embeddings, rank_chunks


def demo_chunk_ranking() -> None:
    """Rank synthetic chunk records against a query vector."""

    print("=" * 70)
    print("DEMO 1 — CHUNK RANKING WITH SYNTHETIC VECTORS")
    print("=" * 70)

    query = "How can a learner reset their password?"
    query_embedding = [1.0, 0.0]
    chunk_records = [
        {
            "text": "Password reset instructions for learner accounts.",
            "metadata": {"source": "account-guide.md", "chunk_index": 0},
            "embedding": [0.98, 0.02],
        },
        {
            "text": "The cafeteria menu changes every Friday.",
            "metadata": {"source": "campus-guide.md", "chunk_index": 3},
            "embedding": [-1.0, 0.0],
        },
        {
            "text": "Learners can recover access using their registered email.",
            "metadata": {"source": "account-guide.md", "chunk_index": 1},
            "embedding": [0.85, 0.15],
        },
    ]

    ranked = rank_chunks(query_embedding, chunk_records)
    print(f"\nQuery: {query}\n")
    for position, record in enumerate(ranked, start=1):
        print(
            f"{position}. score={record['score']:.4f} | "
            f"{record['text']} | {record['metadata']}"
        )


def demo_semantic_similarity() -> None:
    """
    Show that semantically related texts score higher than unrelated ones,
    using synthetic vectors that approximate what a real embedding model
    would produce.

    In production, these vectors come from an API call like:
        embeddings = embed(texts)   # returns real high-dimensional vectors
    Here we use 3-dimensional approximations so the demo runs without an API key.
    """

    print("\n" + "=" * 70)
    print("DEMO 2 — SEMANTIC SIMILARITY (SIMILAR vs DISSIMILAR PAIR)")
    print("=" * 70)

    texts = [
        "How do I reset my account password?",    # index 0 — query
        "Steps to recover access to my login",    # index 1 — similar
        "The cafeteria menu has pasta today",      # index 2 — dissimilar
    ]

    # Synthetic 3-D vectors approximating semantic closeness:
    #   texts[0] and texts[1] are both about account/access → close direction
    #   texts[2] is about food → very different direction
    synthetic_embeddings = [
        [0.9, 0.4, 0.1],   # "reset password"
        [0.85, 0.45, 0.1],  # "account recovery" — nearly same direction
        [0.1, 0.1, 0.99],   # "cafeteria menu" — orthogonal direction
    ]

    similar_score = cosine_similarity(
        synthetic_embeddings[0], synthetic_embeddings[1]
    )
    dissimilar_score = cosine_similarity(
        synthetic_embeddings[0], synthetic_embeddings[2]
    )

    print(f'\nText A (query):      "{texts[0]}"')
    print(f'Text B (similar):    "{texts[1]}"')
    print(f'Text C (dissimilar): "{texts[2]}"')

    print(f"\nA vs B — password vs login recovery:  {similar_score:.6f}")
    print(f"A vs C — password vs cafeteria menu:  {dissimilar_score:.6f}")

    if similar_score > dissimilar_score:
        print(
            "\nResult: similar pair scored HIGHER than dissimilar pair — "
            "exactly what we expect from a well-trained embedding model."
        )
    else:
        print(
            "\nResult: unexpected ordering — check the embedding vectors."
        )

    print("\n--- Using compare_embeddings helper ---")
    results = compare_embeddings(
        query_embedding=synthetic_embeddings[0],
        candidate_embeddings=synthetic_embeddings[1:],
        labels=[texts[1], texts[2]],
    )
    for item in results:
        print(f"  rank {item['rank']} | score={item['score']:.6f} | {item['label']}")

    print(
        "\nKey insight: in a real RAG system these vectors are produced by an "
        "embedding model (e.g. gemini-embedding-001). The same ranking logic "
        "applies at any dimension — the direction of vectors in vector space "
        "represents semantic meaning."
    )


def main() -> None:
    demo_chunk_ranking()
    demo_semantic_similarity()


if __name__ == "__main__":
    main()