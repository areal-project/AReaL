"""Persist the rollout worker's native SGLang log via its supported config API."""

import json
import os
from pathlib import Path

path = Path(os.environ["SGLANG_LOGGING_CONFIG_PATH"])
path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
worker = path.stem
config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "sglang": {
            "class": "sglang.srt.utils.NewLineFormatter",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "format": "%(asctime)s.%(msecs)03d %(levelname)s %(process)d ["
            + worker
            + " %(filename)s:%(lineno)d] %(message)s",
        }
    },
    "handlers": {
        "sglang": {
            "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
            "formatter": "sglang",
            "level": "INFO",
            "filename": str(path.with_suffix(".log")),
            "maxBytes": 104857600,
            "backupCount": 20,
        }
    },
    "loggers": {
        "sglang": {"handlers": ["sglang"], "level": "INFO", "propagate": False},
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
}
path.write_text(json.dumps(config, indent=2) + "\n")
path.chmod(0o600)
print("SGLang native log: " + str(path.with_suffix(".log")), flush=True)
