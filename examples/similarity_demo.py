"""Demonstrate cosine similarity ranking for retrieved chunk records."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.similarity import rank_chunks


def main():
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
    print("query:", query)
    for position, record in enumerate(ranked, start=1):
        print(f"{position}. score={record['score']:.4f} {record['text']} {record['metadata']}")


if __name__ == "__main__":
    main()