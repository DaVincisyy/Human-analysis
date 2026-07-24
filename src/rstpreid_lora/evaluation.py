from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from .metrics import retrieval_metrics, topk_records


def _image_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [row["image"].convert("RGB") for row in batch],
        "ids": [int(row["id"]) for row in batch],
        "paths": [str(row["file_path"]) for row in batch],
    }


def encode_split(
    model,
    processor: CLIPProcessor,
    split: Dataset,
    device: torch.device,
    batch_size: int,
    max_images: int | None = None,
) -> dict[str, Any]:
    if max_images is not None:
        split = split.select(range(min(max_images, len(split))))

    loader = DataLoader(
        split,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_image_collate,
        pin_memory=True,
    )
    image_features: list[torch.Tensor] = []
    image_ids: list[int] = []
    image_paths: list[str] = []
    elapsed_image = 0.0

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch in loader:
            inputs = processor(images=batch["images"], return_tensors="pt").to(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            features = model.get_image_features(**inputs)
            torch.cuda.synchronize(device)
            elapsed_image += time.perf_counter() - start
            features = torch.nn.functional.normalize(features, dim=-1)
            image_features.append(features.cpu())
            image_ids.extend(batch["ids"])
            image_paths.extend(batch["paths"])

    query_texts: list[str] = []
    query_ids: list[int] = []
    for row in split:
        for caption in row["captions"]:
            query_texts.append(caption)
            query_ids.append(int(row["id"]))

    text_features: list[torch.Tensor] = []
    elapsed_text = 0.0
    with torch.inference_mode():
        for start_index in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[start_index : start_index + batch_size]
            inputs = processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            features = model.get_text_features(**inputs)
            torch.cuda.synchronize(device)
            elapsed_text += time.perf_counter() - start
            features = torch.nn.functional.normalize(features, dim=-1)
            text_features.append(features.cpu())

    return {
        "image_features": torch.cat(image_features),
        "image_ids": np.asarray(image_ids),
        "image_paths": image_paths,
        "text_features": torch.cat(text_features),
        "text_ids": np.asarray(query_ids),
        "texts": query_texts,
        "image_latency_ms": 1000.0 * elapsed_image / len(image_ids),
        "text_latency_ms": 1000.0 * elapsed_text / len(query_ids),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def evaluate_model(
    model,
    processor: CLIPProcessor,
    split: Dataset,
    device: torch.device,
    batch_size: int,
    max_images: int | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    encoded = encode_split(
        model=model,
        processor=processor,
        split=split,
        device=device,
        batch_size=batch_size,
        max_images=max_images,
    )
    metrics, ranking = retrieval_metrics(
        query_features=encoded["text_features"],
        gallery_features=encoded["image_features"],
        query_ids=encoded["text_ids"],
        gallery_ids=encoded["image_ids"],
    )
    metrics.update(
        {
            "image_latency_ms": encoded["image_latency_ms"],
            "text_latency_ms": encoded["text_latency_ms"],
            "peak_gpu_memory_mb": encoded["peak_gpu_memory_mb"],
        }
    )
    records = topk_records(
        ranking=ranking,
        query_texts=encoded["texts"],
        query_ids=encoded["text_ids"],
        gallery_paths=encoded["image_paths"],
        gallery_ids=encoded["image_ids"],
    )
    return metrics, records
