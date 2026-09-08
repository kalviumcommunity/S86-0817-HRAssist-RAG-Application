import unittest
from unittest.mock import patch


class TestQueryApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from src.app import app

        self.client = TestClient(app)

    def test_short_question_is_rejected(self):
        response = self.client.post("/query", json={"question": "Hi"})
        self.assertEqual(response.status_code, 422)

    def test_query_returns_structured_answer_and_sources(self):
        import src.app as app_module

        records = [{
            "id": "policy.txt:0",
            "text": "Submit through the HR portal.",
            "embedding": [1.0, 0.0],
            "metadata": {"source": "policy.txt", "chunk_index": 0},
        }]

        with patch.object(app_module, "VECTOR_STORE", records), \
             patch.object(app_module, "_embed_fn", lambda texts: [[1.0, 0.0]]), \
             patch.object(app_module, "_generate_fn", lambda question, context: "Submit online."):
            response = self.client.post(
                "/query",
                json={"question": "How do I submit?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "answer": "Submit online.",
            "sources": [{
                "source": "policy.txt",
                "chunk_id": "policy.txt:0",
                "score": 1.0,
            }],
            "status": "answered",
        })

    def test_unconfigured_query_returns_503(self):
        import src.app as app_module

        with patch.object(app_module, "_embed_fn", None), patch.object(app_module, "_generate_fn", None):
            response = self.client.post(
                "/query",
                json={"question": "How do I submit?"},
            )
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()