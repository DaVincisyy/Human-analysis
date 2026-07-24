from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import CLIPProcessor


def load_rstpreid(path: str | Path) -> DatasetDict:
    dataset = load_from_disk(str(path))
    expected = {"train", "val", "test"}
    if set(dataset) != expected:
        raise ValueError(f"Expected splits {expected}, found {set(dataset)}")
    return dataset


def validate_identity_splits(dataset: DatasetDict) -> dict[str, Any]:
    identity_sets = {
        split: set(map(int, dataset[split]["id"])) for split in dataset.keys()
    }
    path_sets = {
        split: set(map(str, dataset[split]["file_path"])) for split in dataset.keys()
    }
    report: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        caption_lengths = sorted(set(map(len, dataset[split]["captions"])))
        report[split] = {
            "images": len(dataset[split]),
            "identities": len(identity_sets[split]),
            "captions_per_image": caption_lengths,
        }
    report["identity_overlap"] = {
        "train_val": len(identity_sets["train"] & identity_sets["val"]),
        "train_test": len(identity_sets["train"] & identity_sets["test"]),
        "val_test": len(identity_sets["val"] & identity_sets["test"]),
    }
    report["path_overlap"] = {
        "train_val": len(path_sets["train"] & path_sets["val"]),
        "train_test": len(path_sets["train"] & path_sets["test"]),
        "val_test": len(path_sets["val"] & path_sets["test"]),
    }
    if any(report["identity_overlap"].values()) or any(report["path_overlap"].values()):
        raise ValueError(f"Dataset leakage detected: {report}")
    return report


class TrainingPairs(TorchDataset):
    def __init__(self, split: Dataset, max_pairs: int | None = None) -> None:
        self.split = split
        self.indices = [
            (image_index, caption_index)
            for image_index, captions in enumerate(split["captions"])
            for caption_index in range(len(captions))
        ]
        if max_pairs is not None:
            self.indices = self.indices[:max_pairs]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_index, caption_index = self.indices[index]
        row = self.split[image_index]
        return {
            "image": row["image"].convert("RGB"),
            "text": row["captions"][caption_index],
            "identity": int(row["id"]),
        }


@dataclass
class CLIPTrainingCollator:
    processor: CLIPProcessor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.processor(
            text=[item["text"] for item in batch],
            images=[item["image"] for item in batch],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return dict(encoded)
