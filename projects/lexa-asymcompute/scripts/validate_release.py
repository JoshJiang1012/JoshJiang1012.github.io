#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "CHANGELOG.md", "CONTRIBUTING.md",
    "docs/MATHEMATICAL_MODEL.md", "docs/MEASURED_DATA.md", "docs/USE_CASES.md",
    "data/observed/virtual_environment_2026-09-02.json",
]
FORBIDDEN_PUBLIC_NAMES = ("5D" + "9A", "JoshJiang1012" + ".github.io")


def main() -> int:
    checks: list[dict[str, object]] = []
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    checks.append({"name": "required_files", "passed": not missing, "details": missing})

    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "__pycache__", ".venv"} for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".md", ".py", ".toml", ".yml", ".yaml", ".json", ".cff", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for term in FORBIDDEN_PUBLIC_NAMES:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)}: {term}")
    checks.append({"name": "public_naming", "passed": not violations, "details": violations})

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    checks.append({"name": "unit_tests", "passed": run.returncode == 0, "details": run.stderr[-3000:]})
    result = {"classification": "observed_release_validation", "measured": True, "passed": all(item["passed"] for item in checks), "checks": checks}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
