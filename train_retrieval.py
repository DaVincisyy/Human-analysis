from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.rstpreid_lora.data import (
    CLIPTrainingCollator,
    TrainingPairs,
    load_rstpreid,
    validate_identity_splits,
)
from src.rstpreid_lora.evaluation import evaluate_model
from src.rstpreid_lora.modeling import create_lora_model, load_processor
from src.rstpreid_lora.utils import (
    load_config,
    parameter_report,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "config.json")

    dataset = load_rstpreid(config["dataset_path"])
    save_json(validate_identity_splits(dataset), output_dir / "dataset_report.json")
    processor = load_processor(config["model_path"])
    model = create_lora_model(config["model_path"], config).to(device)
    report = parameter_report(model)
    save_json(report, output_dir / "parameters.json")
    print(f"Parameter report: {report}")

    train_config = config["train"]
    train_dataset = TrainingPairs(
        dataset["train"], max_pairs=train_config["max_train_pairs"]
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config["num_workers"]),
        pin_memory=True,
        collate_fn=CLIPTrainingCollator(processor),
    )
    accumulation = int(train_config["gradient_accumulation_steps"])
    epochs = int(train_config["epochs"])
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * float(train_config["warmup_ratio"]))

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    precision = str(train_config["mixed_precision"])
    use_fp16 = precision == "fp16"
    use_mixed_precision = precision in {"fp16", "bf16"}
    autocast_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    history: list[dict[str, float | int]] = []
    best_recall = -1.0
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.amp.autocast(
                "cuda", dtype=autocast_dtype, enabled=use_mixed_precision
            ):
                outputs = model(**batch, return_loss=True)
                loss = outputs.loss / accumulation
            scaler.scale(loss).backward()
            running_loss += float(loss.detach()) * accumulation

            should_update = step % accumulation == 0 or step == len(loader)
            if should_update:
                if use_fp16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if use_fp16:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_ran = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()
                    optimizer_ran = True
                if optimizer_ran:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            if step % int(train_config["log_every"]) == 0:
                progress.set_postfix(loss=running_loss / step)

        val_metrics, _ = evaluate_model(
            model=model,
            processor=processor,
            split=dataset["val"],
            device=device,
            batch_size=int(train_config["eval_batch_size"]),
            max_images=train_config["max_eval_images"],
        )
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(loader),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.perf_counter() - started,
            **val_metrics,
        }
        history.append(record)
        save_json(history, output_dir / "history.json")
        print(f"Epoch {epoch}: {record}")

        if val_metrics["recall_at_1"] > best_recall:
            best_recall = val_metrics["recall_at_1"]
            adapter_dir = output_dir / "best_adapter"
            model.save_pretrained(adapter_dir)
            processor.save_pretrained(adapter_dir)
            save_json(record, output_dir / "best_validation.json")

    print(f"Training complete. Best validation R@1: {best_recall:.2f}")


if __name__ == "__main__":
    main()
