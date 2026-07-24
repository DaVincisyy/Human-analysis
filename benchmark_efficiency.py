from __future__ import annotations

import argparse
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.rstpreid_lora.data import (
    CLIPTrainingCollator,
    TrainingPairs,
    load_rstpreid,
)
from src.rstpreid_lora.modeling import create_lora_model, load_processor
from src.rstpreid_lora.utils import (
    load_config,
    parameter_report,
    resolve_device,
    save_json,
    set_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--output", default="outputs/main_r16_a32_all/training_efficiency.json"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device()
    processor = load_processor(config["model_path"])
    model = create_lora_model(config["model_path"], config).to(device)
    train_pairs = TrainingPairs(
        load_rstpreid(config["dataset_path"])["train"],
        max_pairs=int(config["train"]["batch_size"]) * args.steps,
    )
    loader = DataLoader(
        train_pairs,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=CLIPTrainingCollator(processor),
        pin_memory=True,
    )
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["train"]["learning_rate"]),
    )

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = 0.0
    measured = 0
    for step, batch in enumerate(loader):
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, return_loss=True).loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        if step > 0:
            elapsed += time.perf_counter() - started
            measured += 1

    result = {
        **parameter_report(model),
        "batch_size": int(config["train"]["batch_size"]),
        "precision": config["train"]["mixed_precision"],
        "training_step_ms": 1000.0 * elapsed / measured,
        "samples_per_second": int(config["train"]["batch_size"])
        * measured
        / elapsed,
        "peak_training_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
