from __future__ import annotations

from typing import Any

import numpy as np
import torch


def retrieval_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    top_k: tuple[int, ...] = (1, 5, 10),
) -> tuple[dict[str, float], np.ndarray]:
    similarities = query_features @ gallery_features.T
    ranking = torch.argsort(similarities, dim=1, descending=True).cpu().numpy()

    recalls = {k: 0 for k in top_k}
    average_precisions: list[float] = []
    for query_index, ranked_indices in enumerate(ranking):
        relevant = gallery_ids[ranked_indices] == query_ids[query_index]
        for k in top_k:
            recalls[k] += int(relevant[:k].any())
        relevant_positions = np.flatnonzero(relevant)
        if len(relevant_positions) == 0:
            average_precisions.append(0.0)
            continue
        precisions = np.arange(1, len(relevant_positions) + 1) / (
            relevant_positions + 1
        )
        average_precisions.append(float(precisions.mean()))

    count = len(query_ids)
    metrics = {f"recall_at_{k}": 100.0 * recalls[k] / count for k in top_k}
    metrics["map"] = 100.0 * float(np.mean(average_precisions))
    metrics["num_queries"] = float(count)
    metrics["gallery_size"] = float(len(gallery_ids))
    return metrics, ranking


def topk_records(
    ranking: np.ndarray,
    query_texts: list[str],
    query_ids: np.ndarray,
    gallery_paths: list[str],
    gallery_ids: np.ndarray,
    limit: int = 10,
) -> list[dict[str, Any]]:
    records = []
    for index, ranked_indices in enumerate(ranking):
        selected = ranked_indices[:limit]
        records.append(
            {
                "query": query_texts[index],
                "query_id": int(query_ids[index]),
                "top_paths": [gallery_paths[item] for item in selected],
                "top_ids": [int(gallery_ids[item]) for item in selected],
                "first_correct_rank": next(
                    (
                        rank + 1
                        for rank, item in enumerate(ranked_indices)
                        if gallery_ids[item] == query_ids[index]
                    ),
                    None,
                ),
            }
        )
    return records
