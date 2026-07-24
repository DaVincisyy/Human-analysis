from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import yaml

from src.rstpreid_lora.utils import load_config


EXPERIMENTS = [
    {
        "name": "reference_r16_a32_all",
        "lora": {"rank": 16, "alpha": 32},
    },
    {
        "name": "rank4_a8_all",
        "lora": {"rank": 4, "alpha": 8},
    },
    {
        "name": "rank8_a16_all",
        "lora": {"rank": 8, "alpha": 16},
    },
    {
        "name": "rank32_a64_all",
        "lora": {"rank": 32, "alpha": 64},
    },
    {
        "name": "alpha16_r16_all",
        "lora": {"rank": 16, "alpha": 16},
    },
    {
        "name": "modules_qv_r16_a32",
        "lora": {
            "rank": 16,
            "alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        },
    },
    {
        "name": "data25_r16_a32_all",
        "train": {"max_train_pairs": 9252},
    },
    {
        "name": "data50_r16_a32_all",
        "train": {"max_train_pairs": 18505},
    },
]


def deep_update(target: dict, update: dict) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value


def is_complete(output_dir: Path, expected_epochs: int) -> bool:
    history_path = output_dir / "history.json"
    if not history_path.exists():
        return False
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(history, list)
        and len(history) >= expected_epochs
        and (output_dir / "best_adapter" / "adapter_model.safetensors").exists()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional experiment names to run; completed outputs are skipped.",
    )
    args = parser.parse_args()
    selected_names = set(args.names) if args.names else None

    base = load_config("configs/main.yaml")
    generated_dir = Path("outputs/ablation_configs")
    generated_dir.mkdir(parents=True, exist_ok=True)

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        if selected_names is not None and name not in selected_names:
            continue
        output_dir = Path("outputs/ablations") / name
        if is_complete(output_dir, expected_epochs=3):
            print(f"[skip] {name}: completed result exists")
            continue

        config = copy.deepcopy(base)
        config["output_dir"] = str(output_dir).replace("\\", "/")
        config["train"]["epochs"] = 3
        config["train"]["log_every"] = 200
        overrides = {key: value for key, value in experiment.items() if key != "name"}
        deep_update(config, overrides)

        config_path = generated_dir / f"{name}.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

        print(f"[run] {name}")
        subprocess.run(
            [
                sys.executable,
                "train_retrieval.py",
                "--config",
                str(config_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
