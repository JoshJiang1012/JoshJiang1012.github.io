from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = PROJECT_ROOT / "scripts" / "collect_router_trace.py"


class CollectRouterTraceWrapperTests(unittest.TestCase):
    def test_cpu_moe_profile_is_forwarded_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_binary = root / "llama-router-trace"
            fake_binary.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env python3
                    import json
                    from pathlib import Path
                    import sys

                    args = sys.argv[1:]
                    output = Path(args[args.index('-o') + 1])
                    domain = args[args.index('--domain') + 1]
                    output.write_text(json.dumps({
                        'schema_version': '2.0',
                        'token': 0,
                        'layer': 0,
                        'experts': [0, 1, 2, 3],
                        'domain': domain,
                        'phase': 'decode',
                        'batch_size': 1,
                        'source': 'llama.cpp:ffn_moe_topk',
                    }) + '\\n', encoding='utf-8')
                    """
                ),
                encoding="utf-8",
            )
            os.chmod(fake_binary, 0o755)

            model = root / "model.gguf"
            model.write_bytes(b"test-model")
            prompt = root / "prompt.txt"
            prompt.write_text("Write a harmless unit test.", encoding="utf-8")
            trace = root / "trace.jsonl"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(COLLECTOR),
                    "--binary",
                    str(fake_binary),
                    "--model",
                    str(model),
                    "--output",
                    str(trace),
                    "--prompt-file",
                    str(prompt),
                    "--domain",
                    "coding-python",
                    "--n-predict",
                    "1",
                    "--ctx-size",
                    "128",
                    "--gpu-layers",
                    "99",
                    "--n-cpu-moe",
                    "36",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            manifest = json.loads(
                trace.with_suffix(".jsonl.run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["n_cpu_moe"], 36)
            self.assertIn("--n-cpu-moe", manifest["command"])
            self.assertIn("36", manifest["command"])
            self.assertFalse(manifest["prompt"]["stored"])
            self.assertNotIn("Write a harmless unit test.", json.dumps(manifest))
            self.assertEqual(manifest["trace_audit"]["events"], 1)


if __name__ == "__main__":
    unittest.main()
