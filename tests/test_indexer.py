import unittest

from src.indexer import index_embeddings, spot_check, to_vector_record


class FakeCollection:
    def __init__(self):
        self.records = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for record_id, vector, text, metadata in zip(
            ids, embeddings, documents, metadatas
        ):
            self.records[record_id] = {
                "embedding": vector,
                "document": text,
                "metadata": metadata,
            }

    def count(self):
        return len(self.records)

    def get(self, ids, include):
        record = self.records[ids[0]]
        return {
            "ids": ids,
            "embeddings": [record["embedding"]],
            "documents": [record["document"]],
            "metadatas": [record["metadata"]],
        }


def embedded_chunks():
    return [
        {
            "id": "policy.txt:0",
            "text": "Leave requests require approval.",
            "embedding": [0.1, 0.2],
            "metadata": {
                "source": "policy.txt",
                "chunk_index": 0,
                "section": "Leave",
                "region": "Global",
            },
        },
        {
            "id": "policy.txt:1",
            "text": "Submit requests through the HR portal.",
            "embedding": [0.3, 0.4],
            "metadata": {"source": "policy.txt", "chunk_index": 1},
        },
    ]


class TestIndexer(unittest.TestCase):
    def test_to_vector_record_preserves_required_fields(self):
        record = to_vector_record(embedded_chunks()[0])
        self.assertEqual(record["id"], "policy.txt:0")
        self.assertEqual(record["text"], "Leave requests require approval.")
        self.assertEqual(record["metadata"], {
            "source": "policy.txt",
            "chunk_index": 0,
            "section": "Leave",
        })

    def test_index_count_and_spot_check(self):
        collection = FakeCollection()
        result = index_embeddings(collection, embedded_chunks(), batch_size=1)

        self.assertEqual(result["expected_count"], 2)
        self.assertEqual(result["inserted"], 2)
        self.assertEqual(result["indexed_count"], 2)
        self.assertTrue(result["count_matches"])
        self.assertEqual(result["failures"], [])
        for chunk in embedded_chunks():
            self.assertTrue(all(spot_check(collection, chunk).values()))

    def test_invalid_batch_size_raises(self):
        with self.assertRaises(ValueError):
            index_embeddings(FakeCollection(), embedded_chunks(), batch_size=0)


if __name__ == "__main__":
    unittest.main()