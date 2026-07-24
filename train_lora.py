"""Backward-compatible entry point for the original project command.

The maintained, configurable implementation lives in ``train_retrieval.py``.
"""

from train_retrieval import main


if __name__ == "__main__":
    main()
