from __future__ import annotations

import argparse
from pathlib import Path

from src.rstpreid_lora.data import load_rstpreid, validate_identity_splits
from src.rstpreid_lora.evaluation import evaluate_model
from src.rstpreid_lora.modeling import (
    load_adapter_model,
    load_base_model,
    load_processor,
)
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
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device()
    dataset = load_rstpreid(config["dataset_path"])
    validate_identity_splits(dataset)
    processor = load_processor(config["model_path"])
    if args.adapter:
        model = load_adapter_model(config["model_path"], args.adapter)
        model_name = "lora_clip"
    else:
        model = load_base_model(config["model_path"])
        model_name = "base_clip"
    model.requires_grad_(False)
    model = model.to(device)

    metrics, records = evaluate_model(
        model=model,
        processor=processor,
        split=dataset[args.split],
        device=device,
        batch_size=int(config["train"]["eval_batch_size"]),
        max_images=args.max_images,
    )
    result = {
        "model": model_name,
        "split": args.split,
        "adapter": args.adapter,
        "metrics": metrics,
        "parameters": parameter_report(model),
    }
    output = Path(args.output)
    save_json(result, output)
    save_json(records, output.with_name(output.stem + "_rankings.json"))
    print(result)


if __name__ == "__main__":
    main()
