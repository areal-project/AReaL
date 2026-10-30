# SPDX-License-Identifier: Apache-2.0

"""Store control-provider credentials privately; the key is never echoed."""

import argparse
import getpass
import json
import os
from pathlib import Path


def main():
    from dotenv import dotenv_values

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".config/areal-elastic/openrouter.json",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(
            "Configuration exists; edit it locally or choose another output"
        )
    defaults = dotenv_values(Path(__file__).with_name("openrouter.env.example"))
    key = getpass.getpass("OpenRouter API key (hidden): ").strip()
    if not key:
        raise ValueError("API key cannot be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(
            {
                "base_url": defaults["AREAL_CONTROL_LLM_BASE_URL"],
                "model": defaults["OPENROUTER_MODEL"],
                "api_key": key,
            },
            stream,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
