from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/llama.cpp/d08c7872-router-trace.patch"
MANIFEST = ROOT / "patches/llama.cpp/manifest.json"


class LlamaPatchTests(unittest.TestCase):
    def test_patch_digest_and_pinned_commit(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["upstream_commit"],
            "d08c7872d6ffe3f059f8647840a29aa390413e27",
        )
        self.assertEqual(
            manifest["patch_sha256"],
            hashlib.sha256(PATCH.read_bytes()).hexdigest(),
        )

    def test_patch_adds_minimal_callback_collector(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        required = (
            "add_subdirectory(router-trace)",
            "llama-router-trace",
            "ffn_moe_topk-",
            "ctx_params.cb_eval = router_trace_cb;",
            "llama.cpp:ffn_moe_topk",
            "--prompt-file",
        )
        for marker in required:
            with self.subTest(marker=marker):
                self.assertIn(marker, patch)

    def test_patch_does_not_modify_model_or_graph_sources(self) -> None:
        patch = PATCH.read_text(encoding="utf-8")
        self.assertNotIn("diff --git a/src/models/openai-moe.cpp", patch)
        self.assertNotIn("diff --git a/src/llama-graph.cpp", patch)


if __name__ == "__main__":
    unittest.main()
