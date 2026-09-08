import json
import unittest
from unittest.mock import patch


class TestStreamingApi(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from src.app import app

        self.client = TestClient(app)

    def test_stream_contains_citations_tokens_and_done(self):
        import src.app as app_module

        records = [{
            "id": "policy.txt:0",
            "text": "Submit through the HR portal.",
            "embedding": [1.0, 0.0],
            "metadata": {"source": "policy.txt", "chunk_index": 0},
        }]
        with patch.object(app_module, "VECTOR_STORE", records), \
             patch.object(app_module, "_embed_fn", lambda texts: [[1.0, 0.0]]), \
             patch.object(app_module, "_generate_fn", lambda question, context: "Submit online now."):
            response = self.client.post(
                "/query/stream",
                json={"question": "How do I submit?"},
            )

        self.assertEqual(response.status_code, 200)
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(events[0]["type"], "citations")
        self.assertEqual(events[0]["sources"][0]["label"], "[1]")
        self.assertTrue(any(event["type"] == "token" for event in events))
        self.assertEqual(events[-1]["type"], "done")

    def test_stream_reports_error_without_erasing_partial_contract(self):
        import src.app as app_module

        with patch.object(app_module, "_embed_fn", None), patch.object(app_module, "_generate_fn", None):
            response = self.client.post(
                "/query/stream",
                json={"question": "How do I submit?"},
            )

        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[-1]["type"], "error")


if __name__ == "__main__":
    unittest.main()