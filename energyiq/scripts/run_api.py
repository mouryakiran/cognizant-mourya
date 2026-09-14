"""Start the EnergyIQ API server.

Usage:
    python scripts/run_api.py [--host 127.0.0.1] [--port 8000] [--reload]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

logging.basicConfig(level=logging.INFO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    print("EnergyIQ API -> http://{}:{}/".format(args.host, args.port))
    print("Dashboard   -> http://{}:{}/".format(args.host, args.port))
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()