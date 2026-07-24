from __future__ import annotations

from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import CLIPModel, CLIPProcessor


def load_processor(model_path: str | Path) -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(
        str(model_path), local_files_only=True, use_fast=True
    )


def load_base_model(model_path: str | Path) -> CLIPModel:
    return CLIPModel.from_pretrained(str(model_path), local_files_only=True)


def create_lora_model(model_path: str | Path, config: dict[str, Any]):
    base_model = load_base_model(model_path)
    lora = config["lora"]
    peft_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        target_modules=list(lora["target_modules"]),
        lora_dropout=float(lora["dropout"]),
        bias="none",
    )
    return get_peft_model(base_model, peft_config)


def load_adapter_model(model_path: str | Path, adapter_path: str | Path):
    base_model = load_base_model(model_path)
    return PeftModel.from_pretrained(base_model, str(adapter_path))
