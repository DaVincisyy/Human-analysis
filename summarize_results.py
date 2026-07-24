from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str((Path(".cache") / "matplotlib").resolve())
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    baseline_path = Path("outputs/baseline/base_test.json")
    if baseline_path.exists():
        result = read_json(baseline_path)
        rows.append(
            {
                "experiment": "Base CLIP",
                "source": "test",
                **result["metrics"],
                **result["parameters"],
            }
        )

    main_path = Path("outputs/main_r16_a32_all/test_metrics.json")
    if main_path.exists():
        result = read_json(main_path)
        rows.append(
            {
                "experiment": "LoRA main r16/a32",
                "source": "test",
                **result["metrics"],
                **result["parameters"],
            }
        )

    ablation_root = Path("outputs/ablations")
    if ablation_root.exists():
        for result_path in sorted(ablation_root.glob("*/best_validation.json")):
            history_path = result_path.parent / "history.json"
            if not history_path.exists() or len(read_json(history_path)) < 3:
                continue
            result = read_json(result_path)
            parameters_path = result_path.parent / "parameters.json"
            parameters = (
                read_json(parameters_path) if parameters_path.exists() else {}
            )
            rows.append(
                {
                    "experiment": result_path.parent.name,
                    "source": "validation",
                    **result,
                    **parameters,
                }
            )
    return rows


def main() -> None:
    rows = collect_rows()
    if not rows:
        raise RuntimeError("No experiment results found.")

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(results_dir / "experiment_summary.csv", index=False)

    sns.set_theme(style="whitegrid")
    metric_columns = ["recall_at_1", "recall_at_5", "recall_at_10", "map"]
    available = [column for column in metric_columns if column in frame.columns]
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(13, 11),
        gridspec_kw={"height_ratios": [1, 2]},
    )
    for axis, (source, title) in zip(
        axes,
        [
            ("test", "Main comparison on the held-out test split"),
            ("validation", "Ablations on the validation split (3 epochs)"),
        ],
    ):
        subset = frame[frame["source"] == source]
        melted = subset.melt(
            id_vars=["experiment", "source"],
            value_vars=available,
            var_name="metric",
            value_name="percent",
        )
        chart = sns.barplot(
            data=melted,
            x="experiment",
            y="percent",
            hue="metric",
            ax=axis,
        )
        chart.set_title(title)
        chart.set_xlabel("")
        chart.set_ylabel("Score (%)")
        chart.tick_params(axis="x", rotation=25)
        chart.legend(title="")
    plt.tight_layout()
    plt.savefig(results_dir / "retrieval_metrics.png", dpi=180)
    plt.close(figure)

    if {"adapter_parameters", "recall_at_1"}.issubset(frame.columns):
        ablations = frame[frame["source"] == "validation"].copy()
        ablations["adapter_parameters_m"] = (
            ablations["adapter_parameters"] / 1_000_000
        )
        plt.figure(figsize=(10, 6))
        chart = sns.scatterplot(
            data=ablations,
            x="adapter_parameters_m",
            y="recall_at_1",
            hue="experiment",
            s=100,
        )
        chart.set_title("Validation accuracy versus trainable LoRA parameters")
        chart.set_xlabel("LoRA parameters (millions)")
        chart.set_ylabel("Validation Recall@1 (%)")
        chart.legend(
            title="Ablation",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_vs_parameters.png", dpi=180)
        plt.close()

    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
