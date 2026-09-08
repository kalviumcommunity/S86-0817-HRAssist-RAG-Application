import unittest

from src.retrieval_tuning import choose_best_setting, evaluate_retrieval_settings


def embed_query(texts):
    return [[float(value) for value in text.split()] for text in texts]


CORPUS = [
    {"text": "reset", "embedding": [1.0, 0.0],
     "metadata": {"source": "account-guide.md", "doc_type": "guide"}},
    {"text": "cafeteria", "embedding": [0.0, 1.0],
     "metadata": {"source": "campus-guide.md", "doc_type": "guide"}},
    {"text": "submission", "embedding": [0.7, 0.7],
     "metadata": {"source": "submission-rubric.md", "doc_type": "rubric"}},
]

TEST_QUERIES = [
    {"query": "1.0 0.0", "expected_source": "account-guide.md"},
    {"query": "0.0 1.0", "expected_source": "campus-guide.md"},
    {"query": "0.7 0.7", "expected_source": "submission-rubric.md"},
]


class TestRetrievalTuning(unittest.TestCase):
    def test_compares_settings_and_reports_hit_rate(self):
        summary = evaluate_retrieval_settings(
            TEST_QUERIES,
            [
                {"name": "k1", "k": 1},
                {"name": "filtered", "k": 3, "filter": {"doc_type": "guide"}},
            ],
            CORPUS,
            embed_query,
        )

        self.assertEqual([row["setting"] for row in summary], ["k1", "filtered"])
        self.assertEqual(summary[0]["hit_rate"], 1.0)
        self.assertEqual(summary[1]["hit_rate"], 2 / 3)
        self.assertEqual(len(summary[0]["details"]), len(TEST_QUERIES))

    def test_score_threshold_is_applied(self):
        summary = evaluate_retrieval_settings(
            TEST_QUERIES[:1],
            [{"name": "strict", "k": 3, "min_score": 1.01}],
            CORPUS,
            embed_query,
        )
        self.assertEqual(summary[0]["hit_rate"], 0.0)
        self.assertFalse(summary[0]["details"][0]["hit"])

    def test_choose_best_setting_uses_highest_hit_rate(self):
        summary = [
            {"setting": "baseline", "hit_rate": 0.5},
            {"setting": "filtered", "hit_rate": 1.0},
        ]
        self.assertEqual(choose_best_setting(summary)["setting"], "filtered")


if __name__ == "__main__":
    unittest.main()