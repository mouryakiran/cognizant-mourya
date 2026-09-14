"""Generate the full EnergyIQ dataset (same schema as the real PJM data).

Usage:
    python scripts/generate_data.py [--start 2005-01-01] [--end 2016-12-31]
        [--regions AEP,COMED] [--out data/processed/generated_energy_data.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from energyiq import config
from energyiq.dataset import generate_dataset

logging.basicConfig(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate EnergyIQ dataset")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end", default="2016-12-31")
    p.add_argument("--regions", default=None, help="comma separated region list")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    regions = args.regions.split(",") if args.regions else config.REGIONS
    out = Path(args.out) if args.out else config.GENERATED_DATASET_PATH

    path = generate_dataset(
        start=args.start,
        end=args.end,
        regions=regions,
        out_path=out,
        seed=args.seed,
    )
    print(f"Done. Dataset written to {path}")


if __name__ == "__main__":
    main()