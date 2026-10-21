"""Check Arena control-plane and gateway auth without generating tokens."""

import argparse
import datetime
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def probe(base_url, path, credential, method="GET"):
    if not credential:
        return {"path": path, "status": None, "error": "missing_credential"}
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={
            "Authorization": "Bearer " + credential,
            "Content-Type": "application/json",
        },
        data=b"{}" if method == "POST" else None,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = response.status
    except urllib.error.HTTPError as error:
        status = error.code
    except Exception as error:
        return {"path": path, "status": None, "error": type(error).__name__}
    return {"path": path, "status": status}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    base_url = os.environ["ARENA_OPENAPI_BASE"]
    control = probe(
        base_url, "/openapi/v1/streams?limit=1", os.getenv("ARENA_OPENAPI_TOKEN", "")
    )
    gateway = probe(
        base_url,
        "/api/v1/chat/completions",
        os.getenv("ARENA_LLM_API_KEY", ""),
        method="POST",
    )
    # The real gateway authenticates before validating the deliberately empty
    # request. A 400 here passes auth but does not dispatch model inference.
    passed = control.get("status") == 200 and gateway.get("status") == 400
    report = {
        "checked_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "passed": passed,
        "control_plane": control,
        "gateway": gateway,
        "inference_executed": False,
        "full_rollout_verified": False,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        json.dump(report, handle, indent=2)
    destination.chmod(0o600)
    print("Arena auth preflight: " + ("passed" if passed else "failed"), flush=True)
    if not passed:
        raise SystemExit("Arena authentication failed before GPU worker allocation")


if __name__ == "__main__":
    main()
