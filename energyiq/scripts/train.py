"""Train and persist the EnergyIQ forecasting model.

Usage:
    python scripts/train.py [--dataset path] [--max-rows 300000]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from energyiq.dataset import load_dataset
from energyiq.forecast import ForecastEngine

logging.basicConfig(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="path to pre-processed CSV")
    p.add_argument("--max-rows", type=int, default=300_000,
                   help="max training rows sampled for speed (0 = use all)")
    args = p.parse_args()

    df = load_dataset(args.dataset)
    engine = ForecastEngine()
    metrics = engine.train(df, max_train_rows=args.max_rows or None, save=True)
    print("Training metrics:", metrics)
    print("Model saved to:", engine.model_path)
    print("Meta saved to:", engine.meta_path)


if __name__ == "__main__":
    main()