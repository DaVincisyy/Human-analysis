# RSTPReid LoRA-CLIP Experiments

## Evaluation protocol

- Dataset: RSTPReid
- Train: 18,505 images / 3,701 identities
- Validation: 1,000 images / 200 identities
- Test: 1,000 images / 200 identities
- Captions: two per image
- Leakage control: identity IDs and file paths are disjoint across all splits
- Query: each test caption (2,000 text queries)
- Gallery: all 1,000 test images
- Positive match: any gallery image with the same identity ID as the query
- Metrics: Recall@1/5/10 and identity-level mAP
- Model selection: highest validation Recall@1
- Test policy: the test split is evaluated only after selecting the best
  validation checkpoint

## Main configuration

- Base model: CLIP ViT-B/32
- LoRA rank: 16
- LoRA alpha: 32
- Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`
- LoRA dropout: 0.05
- Precision: BF16
- Batch size: 16
- Gradient accumulation: 2
- Epochs: 5
- Optimizer: AdamW
- Learning rate: 1e-4 with warmup and cosine decay
- Seed: 42

## Main test results

| Model | Recall@1 | Recall@5 | Recall@10 | mAP |
|---|---:|---:|---:|---:|
| Base CLIP | 8.90 | 21.25 | 31.05 | 6.29 |
| LoRA-CLIP | **34.70** | **60.60** | **72.05** | **27.90** |
| Absolute gain | **+25.80** | **+39.35** | **+41.00** | **+21.61** |

## Efficiency

| Item | Measured value |
|---|---:|
| LoRA parameters | 1,966,080 |
| Adapter percentage | 1.283% |
| Adapter file size | 7.89 MB |
| Peak training GPU memory | 1,435 MB |
| Training throughput | 96.9 samples/s |
| Image encoding latency | 1.98 ms/image |
| Text encoding latency | 0.85 ms/query |

Measurements were collected on an NVIDIA GeForce RTX 5060 Laptop GPU with
PyTorch 2.11.0, CUDA 12.8, BF16, and batch size 16.

## Ablation design

All ablations use the same seed, dataset split, optimizer, evaluation code, and
three training epochs. The controlled comparisons cover:

- Rank: 4, 8, 16, 32
- Alpha: 16, 32 at rank 16
- Target modules: `q_proj/v_proj` versus all four attention projections
- Training pairs: 25%, 50%, 100%

The generated configurations and raw logs are stored below `outputs/`. Compact
tables and figures are written to `results/`.

## Ablation results

The table reports each run's best validation checkpoint. These validation
numbers are used only for controlled comparison and are not mixed with the
held-out test results above.

| Comparison | Rank / alpha | Target modules | Train data | LoRA params | Params (%) | R@1 | R@5 | R@10 | mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Rank 4 | 4 / 8 | q/k/v/out | 100% | 491,520 | 0.324 | 24.75 | 45.10 | 55.90 | 20.19 |
| Rank 8 | 8 / 16 | q/k/v/out | 100% | 983,040 | 0.646 | 26.70 | 46.30 | 55.85 | 21.21 |
| Reference | 16 / 32 | q/k/v/out | 100% | 1,966,080 | 1.283 | 26.85 | 48.50 | 58.50 | 21.54 |
| Rank 32 | 32 / 64 | q/k/v/out | 100% | 3,932,160 | 2.533 | 26.70 | 48.35 | 58.30 | 21.80 |
| Alpha 16 | 16 / 16 | q/k/v/out | 100% | 1,966,080 | 1.283 | 26.05 | 48.10 | 59.05 | 21.10 |
| q/v only | 16 / 32 | q/v | 100% | 983,040 | 0.646 | **27.05** | 46.40 | 56.80 | 21.26 |
| Data 25% | 16 / 32 | q/k/v/out | 25% | 1,966,080 | 1.283 | 21.40 | 42.80 | 53.15 | 17.35 |
| Data 50% | 16 / 32 | q/k/v/out | 50% | 1,966,080 | 1.283 | 23.20 | 43.25 | 54.25 | 18.30 |

Key observations:

- The q/v-only adapter gives the highest three-epoch validation R@1 (27.05%)
  with 983,040 parameters, half of the four-module reference.
- Rank 32 uses eight times as many adapter parameters as Rank 4 but improves
  R@1 by only 1.95 points; increasing rank beyond 8--16 has diminishing returns.
- Using 25%, 50%, and 100% of the training pairs yields R@1 values of 21.40%,
  23.20%, and 26.85%, respectively, confirming that additional paired data is
  still beneficial.
- Alpha 32 improves R@1 by 0.80 points over alpha 16 at the same rank and
  parameter count.
