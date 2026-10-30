# SPDX-License-Identifier: Apache-2.0

"""Strict live-worktree SWE grading; infrastructure failures are not rewards."""

import json
import re
import shlex
from pathlib import Path


def validate_grading(
    reward: float, output: str, failures: int, candidate_failure=None
) -> str:
    """A caught SDK/transport failure is not a legitimate negative reward."""
    if failures or reward not in (0.0, 1.0):
        raise RuntimeError("Sandbox infrastructure failed or reward is invalid")
    expected = "AUTOPILOT_SWE_RESOLVED=" + str(bool(reward)).lower()
    other = "AUTOPILOT_SWE_RESOLVED=" + str(not bool(reward)).lower()
    if expected not in output or other in output:
        if reward == 0 and other not in output and candidate_failure:
            return "candidate_syntax_failure"
        raise RuntimeError("Reward lacks an unambiguous completed grading report")
    return "completed_reference_report"


async def compile_snapshot(env, patch):
    """Compare candidate and HEAD under the benchmark's testbed interpreter."""
    paths = []
    for line in patch.splitlines():
        if not line.startswith("+++ "):
            continue
        name = shlex.split(line)[1]
        if name.startswith("b/"):
            name = name[2:]
        path = Path(name)
        if path.suffix == ".py" and not path.is_absolute() and ".." not in path.parts:
            paths.append(name)
    if not paths:
        return None
    code = """import hashlib,json,pathlib,subprocess,sys
result={'python':sys.executable,'files':{}}
for name in sys.argv[1:]:
 try:
  source=pathlib.Path(name).read_bytes()
  baseline=subprocess.check_output(['git','show','HEAD:'+name])
  row={'source_sha256':hashlib.sha256(source).hexdigest(),'baseline_sha256':hashlib.sha256(baseline).hexdigest(),'baseline_compiles':False,'candidate_error':None}
  try:
   compile(baseline,name,'exec');row['baseline_compiles']=True
  except SyntaxError:pass
  try:compile(source,name,'exec')
  except SyntaxError as error:row['candidate_error']={'type':type(error).__name__,'line':error.lineno,'message':error.msg}
  result['files'][name]=row
 except Exception as error:result['files'][name]={'diagnostic_error':str(error)}
print('AUTOPILOT_CANDIDATE_COMPILE='+json.dumps(result,sort_keys=True))
"""
    result = await env.execute_bash(
        "cd "
        + shlex.quote(env.config.repo_dir)
        + " && "
        + shlex.join(["conda", "run", "-n", "testbed", "python", "-c", code, *paths])
    )
    if result["returncode"] == 0:
        for line in result["output"].splitlines():
            if line.startswith("AUTOPILOT_CANDIDATE_COMPILE="):
                return json.loads(line.split("=", 1)[1])
    return None


async def verify_candidate_syntax_failure(env, patch, reward, output, before):
    """Accept only a reproducible candidate error matching the grading traceback."""
    # The SWE-Gym wrapper emits this error only after its native test command
    # returns and the official parser reports missing references. Its scripts
    # do not use the older Verified harness's Start/End Test Output delimiters.
    required = ["Testing result:"]
    if "Evaluation omitted reference tests:" not in output:
        required += [
            "Evaluation did not report every reference test",
            ">>>>> Start Test Output",
            ">>>>> End Test Output",
        ]
    if (
        reward != 0
        or env.tool_call_failed_count
        or not before
        or any(marker not in output for marker in required)
        or "get_reward error" in output
    ):
        return None
    after = await compile_snapshot(env, patch)
    if (
        not after
        or after["python"] != before["python"]
        or "/envs/testbed/" not in after["python"]
    ):
        return None
    for name, prior in before["files"].items():
        current = after["files"].get(name, {})
        error = prior.get("candidate_error")
        if not error or not prior.get("baseline_compiles") or prior != current:
            continue
        if (
            error["type"] not in ("SyntaxError", "IndentationError", "TabError")
            or error["type"] + ":" not in output
        ):
            continue
        if re.search(
            re.escape(name) + r".{0,512}?line\s+" + str(error["line"]) + r"\D",
            output,
            re.S,
        ):
            return {"file": name, "python": after["python"], **prior}
    return None


async def grade_live_candidate(env, patch):
    """The agent already changed the worktree; SDK must not apply it twice."""
    current = await env.submit_final_solution()
    if current["returncode"] != 0 or current["output"] != patch:
        raise RuntimeError(
            "Candidate changed between final patch extraction and grading"
        )
    # Pinned SDK executes eval_script independently of its optional patch apply.
    return await env.request("get_reward", patch="", return_output=True)
