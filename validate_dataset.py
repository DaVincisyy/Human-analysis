from __future__ import annotations

import argparse

from src.rstpreid_lora.data import load_rstpreid, validate_identity_splits
from src.rstpreid_lora.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/rstpreid")
    parser.add_argument("--output", default="outputs/dataset_report.json")
    args = parser.parse_args()
    report = validate_identity_splits(load_rstpreid(args.dataset))
    save_json(report, args.output)
    print(report)


if __name__ == "__main__":
    main()
