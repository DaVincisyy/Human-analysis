from __future__ import annotations

import argparse
import os
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve())
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap

from src.rstpreid_lora.data import load_rstpreid
from src.rstpreid_lora.evaluation import encode_split
from src.rstpreid_lora.metrics import retrieval_metrics
from src.rstpreid_lora.modeling import (
    load_adapter_model,
    load_base_model,
    load_processor,
)
from src.rstpreid_lora.utils import load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def plot_similarity_distribution(
    similarities: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    target: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for index, identity in enumerate(query_ids):
        positives = similarities[index, gallery_ids == identity]
        negatives = similarities[index, gallery_ids != identity]
        positive_scores.extend(positives.tolist())
        sample_size = min(10, len(negatives))
        negative_scores.extend(
            rng.choice(negatives, size=sample_size, replace=False).tolist()
        )
    plt.figure(figsize=(8, 5))
    sns.kdeplot(positive_scores, label="Positive (same identity)", fill=True)
    sns.kdeplot(negative_scores, label="Negative", fill=True)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_umap(
    text_features: np.ndarray,
    image_features: np.ndarray,
    text_ids: np.ndarray,
    image_ids: np.ndarray,
    target: Path,
) -> None:
    selected_ids = np.unique(image_ids)[:20]
    image_mask = np.isin(image_ids, selected_ids)
    text_mask = np.isin(text_ids, selected_ids)
    combined = np.concatenate(
        [image_features[image_mask], text_features[text_mask]], axis=0
    )
    identities = np.concatenate([image_ids[image_mask], text_ids[text_mask]])
    modalities = np.array(
        ["image"] * int(image_mask.sum()) + ["text"] * int(text_mask.sum())
    )
    reducer = umap.UMAP(
        n_components=2, n_neighbors=15, min_dist=0.15, metric="cosine", random_state=42
    )
    projected = reducer.fit_transform(combined)

    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab20", n_colors=len(selected_ids))
    color_map = {identity: palette[index] for index, identity in enumerate(selected_ids)}
    for modality, marker in (("image", "o"), ("text", "X")):
        mask = modalities == modality
        colors = [color_map[identity] for identity in identities[mask]]
        plt.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=colors,
            marker=marker,
            s=36,
            alpha=0.75,
            label=modality,
        )
    plt.legend(title="Modality")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close()


def plot_failure_cases(
    ranking: np.ndarray,
    texts: list[str],
    text_ids: np.ndarray,
    image_ids: np.ndarray,
    image_paths: list[str],
    split,
    target: Path,
) -> None:
    first_correct = []
    for query_index, ranked_indices in enumerate(ranking):
        relevant = np.flatnonzero(image_ids[ranked_indices] == text_ids[query_index])
        first_correct.append(int(relevant[0]) if len(relevant) else len(image_ids))
    selected_queries = np.argsort(first_correct)[-4:][::-1]
    image_lookup = {
        str(row["file_path"]): row["image"].convert("RGB") for row in split
    }

    figure, axes = plt.subplots(
        len(selected_queries),
        6,
        figsize=(18, 12),
        gridspec_kw={"width_ratios": [3.2, 1, 1, 1, 1, 1]},
    )
    for row_index, query_index in enumerate(selected_queries):
        axes[row_index, 0].axis("off")
        axes[row_index, 0].text(
            0.02,
            0.5,
            "Query identity "
            f"{text_ids[query_index]}\n\n"
            + textwrap.fill(texts[query_index], width=42),
            va="center",
            ha="left",
            fontsize=8,
        )
        for column, gallery_index in enumerate(ranking[query_index][:5], start=1):
            path = image_paths[gallery_index]
            axes[row_index, column].imshow(image_lookup[path])
            correct = image_ids[gallery_index] == text_ids[query_index]
            axes[row_index, column].set_title(
                f"Top {column} | id={image_ids[gallery_index]}",
                color="green" if correct else "red",
                fontsize=8,
            )
            axes[row_index, column].axis("off")
    plt.tight_layout()
    plt.savefig(target, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device()
    dataset = load_rstpreid(config["dataset_path"])
    processor = load_processor(config["model_path"])
    if args.adapter:
        model = load_adapter_model(config["model_path"], args.adapter)
    else:
        model = load_base_model(config["model_path"])
    model.requires_grad_(False)
    model = model.to(device)

    encoded = encode_split(
        model=model,
        processor=processor,
        split=dataset["test"],
        device=device,
        batch_size=int(config["train"]["eval_batch_size"]),
    )
    metrics, ranking = retrieval_metrics(
        encoded["text_features"],
        encoded["image_features"],
        encoded["text_ids"],
        encoded["image_ids"],
    )
    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)
    similarities = (
        encoded["text_features"] @ encoded["image_features"].T
    ).numpy()
    plot_similarity_distribution(
        similarities,
        encoded["text_ids"],
        encoded["image_ids"],
        output_dir / "similarity_distribution.png",
        int(config["seed"]),
    )
    plot_umap(
        encoded["text_features"].numpy(),
        encoded["image_features"].numpy(),
        encoded["text_ids"],
        encoded["image_ids"],
        output_dir / "feature_umap.png",
    )
    plot_failure_cases(
        ranking,
        encoded["texts"],
        encoded["text_ids"],
        encoded["image_ids"],
        encoded["image_paths"],
        dataset["test"],
        output_dir / "failure_cases.png",
    )
    print(metrics)


if __name__ == "__main__":
    main()
