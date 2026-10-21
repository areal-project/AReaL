import json
import os
from pathlib import Path

from omegaconf import OmegaConf
from wandb.sdk.artifacts._internal_artifact import InternalArtifact

from examples.swe.utils import SWEPPOConfig

from areal.api.cli_args import to_structured_cfg

root = Path(os.environ["EXPERIMENT_ROOT"])
cfg = OmegaConf.to_object(
    to_structured_cfg(OmegaConf.load(os.environ["QWEN_CONFIG"]), SWEPPOConfig)
)
runid = f"{cfg.stats_logger.experiment_name}_{cfg.stats_logger.trial_name}_{cfg.stats_logger.wandb.id_suffix}"
name = f"run-{runid}-moe_balance/expert_loads"
assert len(name) <= 128
InternalArtifact(name=name, type="run_table")
result = {
    "artifact_name_validation_passed": True,
    "training_run_id": runid,
    "artifact_name": name,
    "length": len(name),
    "extra_wandb_run_created": False,
}
(root / "wandb-preflight-result.json").write_text(json.dumps(result, indent=2))
print("WANDB_NAME_PREFLIGHT_PASS " + json.dumps(result), flush=True)
