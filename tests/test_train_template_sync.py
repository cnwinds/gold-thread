"""训练模板关键部署约定：与外部 goldthread-train 模板包保持同步。"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parents[1] / "pipeline" / "train_template"


class TrainTemplateSyncTests(unittest.TestCase):
    def test_vllm_deploy_guidance_includes_language_model_only_auto_ctx(self):
        deploy = (TEMPLATE / "docs" / "DEPLOY.md").read_text(encoding="utf-8")
        self.assertIn("--language-model-only", deploy)
        self.assertIn("--max-model-len auto", deploy)
        self.assertIn("--gpu-memory-utilization 0.94", deploy)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True", deploy)

    def test_known_issues_registers_vllm_cudagraph_oom(self):
        issues = json.loads((TEMPLATE / "issues" / "known_issues.json").read_text(encoding="utf-8"))
        ids = {item.get("id") for item in issues.get("issues") or []}
        self.assertIn("vllm_cudagraph_oom", ids)
        troubleshooting = (TEMPLATE / "docs" / "TROUBLESHOOTING.md").read_text(encoding="utf-8")
        self.assertIn("vllm_cudagraph_oom", troubleshooting)

    def test_template_documents_package_layout_and_probe(self):
        template_md = (TEMPLATE / "TEMPLATE.md").read_text(encoding="utf-8")
        layout = (TEMPLATE / "docs" / "LAYOUT.md").read_text(encoding="utf-8")
        settings = (TEMPLATE / "settings.yaml").read_text(encoding="utf-8")
        autotrain = (TEMPLATE / "autotrain.py").read_text(encoding="utf-8")
        readme = (TEMPLATE / "README.md").read_text(encoding="utf-8")
        plans = (TEMPLATE / "configs" / "plans.yaml").read_text(encoding="utf-8")
        filt = (TEMPLATE / "tools" / "filter_by_cutoff.py").read_text(encoding="utf-8")
        self.assertIn("packages/", template_md)
        self.assertIn("packages/", layout)
        self.assertIn("probe_before_train", settings)
        self.assertIn("probe_max_peak_gb", settings)
        self.assertIn("train_longest_first", settings)
        self.assertIn("fill_missing_tools_from_sft", settings)
        self.assertIn("/data/work/_template", settings)
        self.assertIn("--no-probe", autotrain)
        self.assertIn("--probe", autotrain)
        self.assertIn("shared_sft_tools", autotrain)
        self.assertIn("dpo_single_offload_10k", autotrain + plans)
        self.assertIn("dpo_single_offload_10k", plans)
        self.assertIn("sft_single_6k", plans)
        self.assertIn("fill_tools", filt)
        self.assertIn("sort_jsonl_by_tokens", filt)
        self.assertIn("TEMPLATE.md", readme)
        self.assertIn("docs/LAYOUT.md", readme)
        self.assertIn("dpo_single_offload", readme)


if __name__ == "__main__":
    unittest.main()
