#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    if args.top_k > args.experts:
        raise SystemExit("top-k cannot exceed experts")
    rng = random.Random(args.seed)
    path = Path(args.output)
    with path.open("w", encoding="utf-8") as handle:
        for token in range(args.tokens):
            for layer in range(args.layers):
                # Biased distribution intentionally exercises cache evaluation.
                weights = [1.0 / (index + 1) for index in range(args.experts)]
                selected: set[int] = set()
                while len(selected) < args.top_k:
                    selected.add(rng.choices(range(args.experts), weights=weights, k=1)[0])
                handle.write(json.dumps({
                    "schema_version": "1.0",
                    "token": token,
                    "layer": layer,
                    "experts": sorted(selected),
                    "phase": "decode",
                    "domain": "synthetic-demo",
                    "source": "lexa-asymcompute-generator",
                    "batch_size": 1
                }) + "\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
