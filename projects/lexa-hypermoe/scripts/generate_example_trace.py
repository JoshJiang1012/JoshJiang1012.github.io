from __future__ import annotations

import json
from pathlib import Path
import sys


def generate(path: Path, *, tokens: int = 10, layers: int = 36) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for token in range(tokens):
            for layer in range(layers):
                base = (token * 7 + layer * 11) % 128
                record = {
                    "schema_version": "1.0",
                    "token": token,
                    "layer": layer,
                    "experts": [
                        base,
                        (base + 17) % 128,
                        (base + 41) % 128,
                        (base + 73) % 128,
                    ],
                    "domain": "synthetic",
                    "phase": "decode",
                    "source": "lexa-hypermoe-example-generator",
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/lexa-hypermoe-example.jsonl")
    generate(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
