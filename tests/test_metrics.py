import unittest

import numpy as np
import torch

from src.rstpreid_lora.metrics import retrieval_metrics


class RetrievalMetricsTests(unittest.TestCase):
    def test_perfect_identity_retrieval(self) -> None:
        gallery = torch.tensor(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]
        )
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        queries = torch.nn.functional.normalize(queries, dim=-1)
        metrics, _ = retrieval_metrics(
            query_features=queries,
            gallery_features=gallery,
            query_ids=np.array([10, 20]),
            gallery_ids=np.array([10, 10, 20, 20]),
        )
        self.assertEqual(metrics["recall_at_1"], 100.0)
        self.assertEqual(metrics["recall_at_5"], 100.0)
        self.assertEqual(metrics["recall_at_10"], 100.0)
        self.assertEqual(metrics["map"], 100.0)

    def test_recall_and_map_use_identity_not_exact_pair(self) -> None:
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.9]])
        gallery = torch.nn.functional.normalize(gallery, dim=-1)
        queries = torch.tensor([[0.0, 1.0]])
        metrics, _ = retrieval_metrics(
            query_features=queries,
            gallery_features=gallery,
            query_ids=np.array([7]),
            gallery_ids=np.array([3, 7, 7]),
        )
        self.assertEqual(metrics["recall_at_1"], 100.0)
        self.assertEqual(metrics["map"], 100.0)


if __name__ == "__main__":
    unittest.main()
