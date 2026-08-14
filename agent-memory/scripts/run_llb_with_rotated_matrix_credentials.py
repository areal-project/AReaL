#!/usr/bin/env python3
"""Run run_llb.py with Matrix credentials injected only in memory.

This wrapper never prints credentials. It also redacts llm/embedding api_key in
run_llb's resolved-config log, whose implementation otherwise serializes the
entire config.
"""
from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


def install_rotated_matrix_credentials() -> None:
    import yaml
    from memrl.configs.config import MempConfig

    config_path = Path(os.environ.get(
        "MEMRL_MATRIX_CREDENTIAL_CONFIG",
        "/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml",
    ))
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mappings = {
        str(item.get("model_name")): (item.get("litellm_params") or {})
        for item in payload.get("model_list", [])
        if isinstance(item, dict)
    }

    def resolve(names: tuple[str, ...]) -> str:
        for name in names:
            value = mappings.get(name, {}).get("api_key")
            if isinstance(value, str) and value.startswith("os.environ/"):
                value = os.environ.get(value.split("/", 1)[1])
            if value:
                return str(value)
        raise RuntimeError(f"No configured Matrix credential for aliases: {names}")

    # Preflighted with gpt-4.1-mini-2025-04-14; never log either value.
    chat_key = resolve(("gpt-4o-2024-11-20", "gpt-4o", "gpt-5-mini"))
    embed_key = resolve(("text-embedding-3-large", "text-embedding-3-small"))

    original_from_yaml = MempConfig.from_yaml.__func__

    @classmethod
    def from_yaml_with_rotated_credentials(cls, config_file):
        config = original_from_yaml(cls, config_file)
        config.llm.api_key = chat_key
        config.embedding.api_key = embed_key
        return config

    def redacted_model_dump_json(self, *args, **kwargs):
        data = self.model_dump(mode="json")
        for section in ("llm", "embedding"):
            if isinstance(data.get(section), dict) and "api_key" in data[section]:
                data[section]["api_key"] = "[REDACTED]"
        return json.dumps(data, ensure_ascii=False, indent=kwargs.get("indent"))

    MempConfig.from_yaml = from_yaml_with_rotated_credentials
    MempConfig.model_dump_json = redacted_model_dump_json
    print("[Matrix] runtime chat+embedding credentials installed (values redacted)", flush=True)


install_rotated_matrix_credentials()

# For checkpoint auditing, execute exactly one validation pass and skip the
# runner's training loop. The checkpoint is supplied by the launcher as a
# node-local copy; no section checkpoint, validation marker, or results file is
# written back to the permanent experiment directory.
_eval_only_section = os.environ.get("MEMRL_EVAL_ONLY_SECTION", "").strip()
if _eval_only_section:
    from memrl.run.llb_rl_runner import LLBRunner

    _section = int(_eval_only_section)

    def _eval_only_run(self):
        if not self.valid_dataset:
            raise RuntimeError("eval-only requested but validation dataset is unavailable")
        print(f"[READONLY_EVAL] starting validation only after Section {_section}", flush=True)
        self._evaluate(self.valid_dataset, "Validation", _section)
        try:
            self.writer.close()
        except Exception:
            pass
        print(f"[READONLY_EVAL] complete after Section {_section}", flush=True)

    LLBRunner.run = _eval_only_run

runpy.run_path(str(Path(__file__).resolve().parents[1] / "run" / "run_llb.py"), run_name="__main__")
