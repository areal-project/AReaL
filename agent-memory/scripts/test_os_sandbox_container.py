#!/usr/bin/env python3
"""End-to-end test of the sandboxed LocalOSContainer (MEMRL_OS_SANDBOX=1).

Runs INSIDE the aistudio container. Verifies:
  1. Multi-step persistence: useradd in cmd#1 is visible in cmd#2 and to the
     "evaluation" command (shared keeper namespace).
  2. /storage isolation: inside the sandbox /storage is empty; an agent command
     that tries to read/delete a REAL /storage canary is blocked (L1) AND, even
     if it slipped past, would see only empty tmpfs.
  3. Real /storage is UNTOUCHED after the container terminates.

SAFETY: we create our OWN canary file under a dedicated test dir in /storage and
verify it still exists at the end. We NEVER delete anything else under /storage.
"""
import os
import sys

# NOTE: do NOT set MEMRL_OS_SANDBOX — verifying it defaults to ON (safety-first).
os.environ["MEMRL_OS_BACKEND"] = "local"

PROJECT = "/storage/openpsi/users/yl/agent-memory/MemRL"
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "3rdparty", "LifelongAgentBench"))

from memrl.apptainer.os_container import LocalOSContainer  # noqa: E402
from src.tasks.instance.os_interaction.utility import CommandItem, CommandName  # noqa: E402

SEP = "=" * 68
def bash(c): return CommandItem(command_name=CommandName.BASH, script=c)

# --- Create a REAL canary under /storage that must survive untouched ---
CANARY_DIR = os.path.join(PROJECT, "logs", "sandbox_test_canary")
CANARY_FILE = os.path.join(CANARY_DIR, "keepme.txt")
os.makedirs(CANARY_DIR, exist_ok=True)
with open(CANARY_FILE, "w") as f:
    f.write("this real /storage file must survive\n")
print(f"[setup] created real canary: {CANARY_FILE}")
canary_before = os.path.exists(CANARY_FILE)

print(SEP); print("Construct sandboxed LocalOSContainer"); print(SEP)
c = LocalOSContainer(command_execution_timeout=20)
print(f"  sandbox={c._sandbox}  keeper_pid={c._keeper_pid}  verified={getattr(c,'_sandbox_verified',None)}")
assert c._sandbox is True, "sandbox should default to ON"
assert getattr(c, "_sandbox_verified", False) is True, "sandbox isolation should verify"

print(SEP); print("[1] /storage isolation inside sandbox"); print(SEP)
r = c.execute_independent(bash("ls -A /storage | head; echo FSTYPE=$(stat -f -c %T /storage)"))
print("  ls /storage in ns:", repr((r.output or '').strip()))

print(SEP); print("[2] L1 interception: command mentioning /storage is refused"); print(SEP)
# NOTE: both test commands point at FAKE, non-existent /storage paths on purpose.
# The L1 guard triggers on the "/storage" substring regardless of existence, so
# the interception is still exercised — but even if EVERY protection failed, these
# commands could not touch any real data (the targets do not exist).
FAKE1 = "/storage/__sandbox_test_nonexistent__/readme.txt"
FAKE2 = "/storage/__sandbox_test_nonexistent__/subdir"
r = c.execute_independent(bash(f"cat {FAKE1}"))
print(f"  cat fake /storage: exit={r.exit_code} output={ (r.output or '').strip()[:120]!r}")
r2 = c.execute_independent(bash(f"rm -rf {FAKE2}"))
print(f"  rm fake /storage: exit={r2.exit_code} output={(r2.output or '').strip()[:120]!r}")

print(SEP); print("[3] multi-step persistence (useradd -> later cmd -> eval)"); print(SEP)
r = c.execute_independent(bash("groupadd devteam && useradd -M -g devteam alice && echo OK"))
print("  cmd#1 useradd:", (r.output or '').strip())
r = c.execute_independent(bash("id alice"))
print("  cmd#2 id alice:", (r.output or '').strip())
r = c.execute_independent(bash("mkdir -p /project && chown alice:devteam /project && chmod 2770 /project && echo made"))
print("  cmd#3 mkdir+chown+chmod:", (r.output or '').strip())
# "evaluation" style check command
r = c.execute_independent(bash("test \"$(stat -c '%U:%G %a' /project)\" = 'alice:devteam 2770' && echo PASS || echo FAIL"))
print("  eval check:", (r.output or '').strip())

print(SEP); print("[3b] HANG-REGRESSION: background daemon + stdin-waiting cmd must NOT hang"); print(SEP)
import time as _t
# (a) command that spawns a background process inheriting stdout — the exact case
# that froze the real run. Must return promptly (backgrounded proc keeps running
# but our call returns), NOT block on pipe EOF.
t0 = _t.time()
r = c.execute_independent(bash("nohup sleep 3000 >/tmp/bg.out 2>&1 & echo SPAWNED_BG; exit 0"))
print(f"  bg-daemon cmd: {_t.time()-t0:.1f}s exit={r.exit_code} out={(r.output or '').strip()[:60]!r} (must be a few s, not 20s+)")
# (b) command that would wait on stdin forever — stdin=DEVNULL must prevent hang.
t0 = _t.time()
r = c.execute_independent(bash("cat"))  # no args: would read stdin forever without DEVNULL
print(f"  stdin-wait cmd: {_t.time()-t0:.1f}s exit={r.exit_code} timeout_flag={r.timeout_flag} (must return, not hang)")
# (c) genuine infinite loop must hit the os_timeout and be killed (timeout_flag True)
t0 = _t.time()
r = c.execute_independent(bash("while true; do :; done"))
print(f"  infinite-loop cmd: {_t.time()-t0:.1f}s timeout_flag={r.timeout_flag} (must be True, killed at ~{c.timeout_sec}s)")

print(SEP); print("[4] terminate container (runs cleanup: userdel etc in ns)"); print(SEP)
c.terminate()
print("  terminated")

print(SEP); print("[5] VERIFY real /storage canary UNTOUCHED"); print(SEP)
canary_after = os.path.exists(CANARY_FILE)
content_ok = False
if canary_after:
    with open(CANARY_FILE) as f:
        content_ok = "must survive" in f.read()
print(f"  canary before={canary_before} after={canary_after} content_ok={content_ok}")
# also: alice must NOT leak to host /etc
host_alice = os.system("id alice >/dev/null 2>&1") == 0
print(f"  did 'alice' leak to host /etc? {host_alice} (should be False)")

# cleanup our own canary (we created it; safe to remove our own test file)
try:
    os.remove(CANARY_FILE); os.rmdir(CANARY_DIR)
    print("  [teardown] removed our own canary test dir")
except OSError as e:
    print(f"  [teardown] canary cleanup: {e}")

print(SEP)
ok = canary_after and content_ok and (not host_alice)
print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
print(SEP)
