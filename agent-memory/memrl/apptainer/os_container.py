"""Container adapters for OS Interaction tasks.

Provides two backends:
- LocalOSContainer: executes commands directly via subprocess in an isolated tmpdir
  (works anywhere with bash/python3/g++/gcc)
- ApptainerOSContainer: uses `apptainer instance` for full container isolation
  (requires Apptainer + loop devices + SIF image)

The default is LocalOSContainer since it works in most environments.
Set MEMRL_OS_BACKEND=apptainer to force the Apptainer path.
"""

import atexit
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Optional

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent.parent
_llb_root = _project_root / "3rdparty" / "LifelongAgentBench"
if str(_llb_root) not in sys.path:
    sys.path.insert(0, str(_llb_root))

# Python 3.10 compat shims (LLB code uses StrEnum which is 3.11+)
import enum as _enum
if not hasattr(_enum, "StrEnum"):
    class _StrEnum(str, _enum.Enum):
        pass
    _enum.StrEnum = _StrEnum  # type: ignore[attr-defined]

import typing as _typing
if not hasattr(_typing, "Self"):
    _typing.Self = object  # type: ignore[attr-defined]
if not hasattr(_typing, "reveal_type"):
    def _noop_reveal_type(x):
        return x
    _typing.reveal_type = _noop_reveal_type  # type: ignore[attr-defined]

from src.tasks.instance.os_interaction.utility import (
    CommandItem,
    CommandName,
    CommandExecutionResult,
)

logger = logging.getLogger(__name__)


import re as _re

# L1 guard: reject any agent command that references a protected mount (default
# /storage). Even inside the ns sandbox /storage is an empty tmpfs, but this is a
# cheap outer guard so we never even attempt such a command.
_PROTECTED_PATH_RE = _re.compile(r'(^|[^A-Za-z0-9_])/storage([/\s"\';]|$)')


def _os_sandbox_enabled() -> bool:
    """Whether to wrap agent commands in an unshare mount-namespace sandbox.

    DEFAULT ON (safety-first): agent bash always runs inside an unshare --mount
    namespace that tmpfs-covers /storage (and /etc), so agent commands can never
    touch the real /storage. Explicitly DISABLE only by setting MEMRL_OS_SANDBOX
    to one of {0,false,no,off} — e.g. for a local env with no /storage mounted
    where you know it's safe. Any other value (or unset) keeps the sandbox ON.
    """
    val = os.environ.get("MEMRL_OS_SANDBOX", "").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def _sandbox_protected_paths() -> list:
    """Paths to hide (tmpfs-cover) inside the sandbox ns. /storage always; extra
    colon-separated paths via MEMRL_OS_SANDBOX_PROTECT."""
    paths = ["/storage"]
    extra = os.environ.get("MEMRL_OS_SANDBOX_PROTECT", "").strip()
    if extra:
        paths += [p for p in extra.split(":") if p.startswith("/")]
    # de-dup, keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _command_references_protected(text: str) -> bool:
    """L1 check: does the command text reference a protected path (e.g. /storage)?"""
    if not text:
        return False
    for p in _sandbox_protected_paths():
        pat = _re.compile(r'(^|[^A-Za-z0-9_])' + _re.escape(p) + r'([/\s"\';]|$)')
        if pat.search(text):
            return True
    return False


# Track live sandbox keeper PIDs so a crash/exit doesn't leave orphaned
# `sleep` processes holding mount namespaces.
_active_keepers: set = set()


def _reap_keepers():
    for pid in list(_active_keepers):
        try:
            os.kill(pid, 9)
        except Exception:
            pass
    _active_keepers.clear()


atexit.register(_reap_keepers)


class LocalOSContainer:
    """Subprocess-based container for OS Interaction tasks.

    Runs commands directly on the host. Since LLB creates a fresh container per
    sample, we clean up common task artifacts on init and terminate.

    For full isolation, run inside Singularity with --writable-tmpfs so that all
    writes to the root filesystem are ephemeral. Without writable-tmpfs, commands
    that write to absolute paths (e.g. /var/log/app) will modify the real host.
    """

    # Non-system top-level paths that LLB tasks may create. Cleaned between samples.
    _CLEANUP_PATHS = [
        "/app", "/appdata", "/backup", "/collab", "/config",
        "/data", "/docs", "/output", "/project", "/project1",
        "/projects", "/reports", "/secure", "/secure_data",
        "/securedocs", "/shared", "/shared_docs", "/shared_resources",
        "/source", "/source_dir", "/src", "/task", "/templates",
        "/test", "/test_dir", "/test_logs", "/testdir", "/workspace",
        "/target", "/devprojects",
        # Subdirs in system paths that tasks create
        "/var/log/app", "/var/www", "/var/report", "/var/dev",
        "/opt/app",
    ]

    def __init__(
        self,
        command_execution_timeout: int,
        image: str = "local-os/default",
    ):
        self.timeout_sec: float = command_execution_timeout
        self._workdir = tempfile.mkdtemp(prefix="llb_os_")
        # Sandbox: a persistent unshare --mount namespace shared by ALL commands
        # of this task (so useradd in one step persists to later steps + eval).
        self._sandbox = _os_sandbox_enabled()
        self._keeper_proc = None
        self._keeper_pid = None
        self._sandbox_verified = False
        if self._sandbox:
            self._start_keeper()
        self._ensure_compat_commands()
        self._cleanup_previous()
        logger.debug("LocalOSContainer initialized: %s (sandbox=%s)", self._workdir, self._sandbox)

    def _start_keeper(self) -> None:
        """Start a persistent mount-namespace keeper that covers protected paths.

        The keeper runs `unshare --mount`, makes mounts private, tmpfs-covers each
        protected path (e.g. /storage) and the whole /etc (repopulated from a copy
        so useradd/userdel only touch the private copy), signals readiness, then
        sleeps to hold the namespace open. All task commands `nsenter` into it.
        """
        ready = os.path.join(self._workdir, "keeper_ready")
        try:
            os.unlink(ready)
        except OSError:
            pass

        cover_cmds = ["mount --make-rprivate / 2>/dev/null || true"]
        for p in _sandbox_protected_paths():
            # cover each protected path with an empty tmpfs (real mount stays under)
            cover_cmds.append(f"mkdir -p {p} 2>/dev/null || true")
            cover_cmds.append(f"mount -t tmpfs -o size=16m tmpfs {p} 2>/dev/null || true")
        # tmpfs-cover the WHOLE /etc then repopulate from a copy (bind-mounting
        # single files breaks useradd/groupadd's rename()-based rewrite).
        etc_bak = os.path.join(self._workdir, "etc_bak")
        cover_cmds += [
            f"cp -a /etc {etc_bak} 2>/dev/null || true",
            "mount -t tmpfs -o size=64m tmpfs /etc 2>/dev/null || true",
            f"cp -a {etc_bak}/. /etc/ 2>/dev/null || true",
        ]
        script = (
            "set +e\n"
            + "\n".join(cover_cmds)
            + f"\ntouch {ready}\n"
            # Bounded lifetime: if this keeper is ever orphaned (parent crash/hang),
            # it self-terminates instead of holding a mount namespace forever. 2h is
            # far longer than any single sample yet guarantees no permanent leak.
            + "exec sleep 7200\n"
        )
        self._keeper_proc = subprocess.Popen(
            [shutil.which("unshare") or "/usr/bin/unshare",
             "--mount", "--propagation", "private", "bash", "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for readiness (covers done).
        for _ in range(100):
            if os.path.exists(ready):
                break
            time.sleep(0.1)
        # The bash keeper exec's sleep; the pid we nsenter into is the keeper proc
        # itself (exec replaces bash with sleep, same pid).
        self._keeper_pid = self._keeper_proc.pid
        if not os.path.exists(ready):
            logger.error("OS sandbox keeper failed to signal readiness; commands will FAIL closed.")
            self._sandbox_verified = False
            return
        _active_keepers.add(self._keeper_pid)
        # VERIFY isolation actually took effect before trusting the sandbox: inside
        # the keeper ns, every protected path must be an EMPTY tmpfs. If not, we do
        # NOT trust it and fail closed (never risk the real protected path).
        self._sandbox_verified = self._verify_isolation()
        logger.info("OS sandbox keeper ready (pid=%s), protected=%s, verified=%s",
                    self._keeper_pid, _sandbox_protected_paths(), self._sandbox_verified)

    def _verify_isolation(self) -> bool:
        """Confirm each protected path is an empty tmpfs inside the keeper ns."""
        nsenter = shutil.which("nsenter") or "/usr/bin/nsenter"
        for p in _sandbox_protected_paths():
            try:
                r = subprocess.run(
                    [nsenter, "--mount", "--target", str(self._keeper_pid), "--",
                     "bash", "-c", f'echo "$(stat -f -c %T {p} 2>/dev/null):$(ls -A {p} 2>/dev/null | head -1)"'],
                    capture_output=True, text=True, timeout=10, stdin=subprocess.DEVNULL,
                )
            except Exception as e:
                logger.error("OS sandbox verify failed for %s: %s", p, e)
                return False
            out = (r.stdout or "").strip()
            fstype, _, first_entry = out.partition(":")
            if fstype != "tmpfs" or first_entry != "":
                logger.error("OS sandbox verify: %s not empty tmpfs (got %r) — FAIL CLOSED", p, out)
                return False
        return True


    def _wrap_cmd(self, cmd: list) -> list:
        """When sandboxed, run `cmd` inside the keeper's mount namespace via nsenter.

        FAIL-CLOSED: sandbox is ON by default. If the keeper isn't ready OR isolation
        could not be verified (protected paths not confirmed empty tmpfs), we RAISE
        rather than silently run on the host — never risk the real /storage."""
        if not self._sandbox:
            return cmd
        if not self._keeper_pid:
            raise RuntimeError("OS sandbox is ON but keeper namespace failed to start; "
                               "refusing to run agent commands on the host. Set "
                               "MEMRL_OS_SANDBOX=0 only in an env with no real /storage.")
        if not getattr(self, "_sandbox_verified", False):
            raise RuntimeError("OS sandbox is ON but isolation of protected paths could not "
                               "be verified; refusing to run agent commands (fail-closed).")
        nsenter = shutil.which("nsenter") or "/usr/bin/nsenter"
        return [nsenter, "--mount", "--target", str(self._keeper_pid), "--", *cmd]

    def _ns_shell(self, script: str, timeout: float = 15.0):
        """Run a bash script (inside the ns when sandboxed) with a hard timeout that
        KILLS the process group on expiry. Redirects output to a FILE (not a PIPE) so
        a backgrounded grandchild that inherits the fd can't block us on pipe EOF.
        Returns the output string, or None on failure/timeout; never blocks forever."""
        import tempfile as _tf
        run_cmd = self._wrap_cmd(["bash", "-c", script])
        try:
            outf = _tf.TemporaryFile(mode="w+", dir=getattr(self, "_workdir", None))
        except Exception:
            outf = _tf.TemporaryFile(mode="w+")

        def _read():
            try:
                outf.seek(0); return outf.read(200000)
            except Exception:
                return None
        try:
            proc = subprocess.Popen(
                run_cmd, stdout=outf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, start_new_session=True,
            )
        except Exception as e:
            logger.warning("_ns_shell exec failed: %s", e)
            try: outf.close()
            except Exception: pass
            return None
        try:
            proc.wait(timeout=timeout)
            out = _read(); outf.close()
            return out
        except subprocess.TimeoutExpired:
            logger.warning("_ns_shell timed out after %ss; killing process group", timeout)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=5)
                    break
                except Exception:
                    continue
            out = _read()
            try: outf.close()
            except Exception: pass
            return out

    def _ensure_compat_commands(self) -> None:
        """Create compatibility symlinks/scripts for Debian/Ubuntu commands on CentOS/RHEL.

        When sandboxed, these run INSIDE the keeper ns so /etc edits (www-data user)
        hit the private /etc copy, not the host."""
        if self._sandbox:
            # Do the setup inside the ns via a single shell script.
            script = r'''
set +e
[ ! -e /usr/sbin/addgroup ] && [ -e /usr/sbin/groupadd ] && ln -s /usr/sbin/groupadd /usr/sbin/addgroup 2>/dev/null
if [ ! -e /usr/local/bin/apt-get ]; then
  printf '#!/bin/bash\n# apt-get shim: silently succeed\nexit 0\n' > /usr/local/bin/apt-get 2>/dev/null && chmod 755 /usr/local/bin/apt-get 2>/dev/null
fi
if ! id www-data >/dev/null 2>&1; then
  groupadd -f www-data 2>/dev/null
  useradd -r -g www-data -s /sbin/nologin www-data 2>/dev/null
fi
'''
            try:
                self._ns_shell(script, timeout=15.0)
            except Exception as e:
                logger.warning("sandbox _ensure_compat_commands failed: %s", e)
            return

        # --- Non-sandbox (host) path: original behavior ---
        # Simple symlinks
        compat_symlinks = {
            "/usr/sbin/addgroup": "/usr/sbin/groupadd",
        }
        for link, target in compat_symlinks.items():
            if not os.path.exists(link) and os.path.exists(target):
                try:
                    os.symlink(target, link)
                except OSError:
                    pass

        # apt-get shim (many LLB tasks use apt-get install which is Debian-only)
        apt_shim = "/usr/local/bin/apt-get"
        if not os.path.exists(apt_shim):
            try:
                with open(apt_shim, "w") as f:
                    f.write("#!/bin/bash\n# apt-get shim for CentOS: silently succeed\nexit 0\n")
                os.chmod(apt_shim, 0o755)
            except OSError:
                pass

        # Ensure www-data user/group exists (Debian convention used by some LLB tasks)
        import subprocess as _sp
        try:
            _sp.run(["id", "www-data"], capture_output=True, timeout=3)
        except Exception:
            pass
        else:
            result = _sp.run(["id", "www-data"], capture_output=True, timeout=3)
            if result.returncode != 0:
                _sp.run(["groupadd", "-f", "www-data"], capture_output=True, timeout=3)
                _sp.run(["useradd", "-r", "-g", "www-data", "-s", "/sbin/nologin", "www-data"],
                        capture_output=True, timeout=3)

    def _cleanup_previous(self) -> None:
        """Remove common LLB task artifacts from previous samples.

        When sandboxed, cleanup runs INSIDE the keeper ns so user/group deletions
        hit the private /etc copy (never the host). Protected paths (/storage) are
        empty tmpfs in the ns, so they are never affected."""
        if self._sandbox:
            if not self._keeper_pid or not getattr(self, "_sandbox_verified", False):
                # Fail-closed: no verified ns => do NOT run destructive cleanup on host.
                logger.warning("sandbox on but keeper not verified; skipping cleanup (fail-closed)")
                return
            paths = " ".join(self._CLEANUP_PATHS)
            keep_u = "nobody nfsnobody systemd-coredump polkitd www-data systemd-network"
            keep_g = "nobody nfsnobody nogroup systemd-coredump polkitd www-data systemd-network"
            # CRITICAL: use `userdel` WITHOUT `-r`. `-r` deletes the user's HOME dir;
            # some container users (e.g. systemd-network/polkitd) have home=/ with
            # UID>=500, so `userdel -r` becomes `rm -rf /` and destroys the container.
            # OS tasks only care whether a user exists in /etc/passwd, not its home,
            # so removing just the account entry is correct and safe. We also skip any
            # user whose home is /, empty, or a system-critical path as belt-and-braces.
            script = f'''
set +e
for p in {paths}; do rm -rf "$p" 2>/dev/null; done
for e in /var/log/*; do [ -L "$e" ] && rm -f "$e" 2>/dev/null; done
while IFS=: read -r name _ uid _ _ home _; do
  [ "$uid" -ge 500 ] 2>/dev/null || continue
  [ "$uid" -lt 60000 ] 2>/dev/null || continue
  case " {keep_u} " in *" $name "*) continue ;; esac
  case "$home" in /|""|/root|/usr|/bin|/sbin|/etc|/var|/lib|/lib64) continue ;; esac
  userdel "$name" 2>/dev/null
done < /etc/passwd
for g in $(awk -F: '$3>=500 && $3<60000 {{print $1}}' /etc/group); do
  case " {keep_g} " in *" $g "*) : ;; *) groupdel "$g" 2>/dev/null ;; esac
done
'''
            try:
                self._ns_shell(script, timeout=30.0)
            except Exception as e:
                logger.warning("sandbox cleanup failed: %s", e)
            return

        # --- Non-sandbox (host) path: original behavior ---
        for path in self._CLEANUP_PATHS:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)

        # Clean dangling symlinks in /var/log (left by previous samples)
        try:
            for entry in os.scandir("/var/log"):
                if entry.is_symlink():
                    os.unlink(entry.path)
        except OSError:
            pass

        # Remove users/groups created by previous samples
        self._cleanup_users_and_groups()

    def _cleanup_users_and_groups(self) -> None:
        """Remove non-system users and groups created by LLB tasks.

        CRITICAL: never use `userdel -r`. `-r` deletes the user's HOME directory,
        and some users (e.g. systemd-network/polkitd) have home=/ with UID>=500, so
        `userdel -r` becomes `rm -rf /`. OS tasks only care whether a user exists in
        /etc/passwd, so we remove just the account entry. We also skip users whose
        home is / or a system-critical path.
        """
        import subprocess as _sp
        _skip_homes = {"/", "", "/root", "/usr", "/bin", "/sbin", "/etc", "/var", "/lib", "/lib64"}
        _keep_users = {"nobody", "nfsnobody", "systemd-coredump", "polkitd",
                       "www-data", "systemd-network"}
        try:
            # Read name + uid + home so we can skip dangerous home dirs.
            result = _sp.run(
                ["awk", "-F:", "$3 >= 500 && $3 < 60000 {print $1\":\"$6}", "/etc/passwd"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                user, home = line.split(":", 1)
                if not user or user in _keep_users:
                    continue
                if home in _skip_homes:
                    continue
                _sp.run(["userdel", user], capture_output=True, timeout=5)  # NO -r
        except Exception:
            pass

        # Get all groups with GID >= 500 (covers 'admin' GID=500 on CentOS)
        try:
            result = _sp.run(
                ["awk", "-F:", "$3 >= 500 && $3 < 60000 {print $1}", "/etc/group"],
                capture_output=True, text=True, timeout=5,
            )
            _keep_groups = {"nobody", "nfsnobody", "nogroup", "systemd-coredump", "polkitd"}
            for group in result.stdout.strip().split("\n"):
                group = group.strip()
                if group and group not in _keep_groups:
                    _sp.run(["groupdel", group], capture_output=True, timeout=5)
        except Exception:
            pass

    def terminate(self) -> None:
        # In sandbox mode, the keeper holds an ephemeral mount namespace: its
        # tmpfs-covered /storage and its private /etc copy vanish the instant the
        # keeper dies. So we KILL THE KEEPER FIRST (never let a hung cleanup leak
        # it) and skip host cleanup entirely — there is nothing persistent to clean.
        keeper = getattr(self, "_keeper_proc", None)
        if getattr(self, "_sandbox", False):
            if keeper is not None:
                pid = keeper.pid
                try:
                    os.killpg(pid, signal.SIGKILL)
                except Exception:
                    try:
                        keeper.kill()
                    except Exception:
                        pass
                try:
                    keeper.wait(timeout=5)
                except Exception:
                    pass
                _active_keepers.discard(pid)
                self._keeper_proc = None
                self._keeper_pid = None
            if os.path.exists(self._workdir):
                shutil.rmtree(self._workdir, ignore_errors=True)
            logger.debug("LocalOSContainer terminated (sandbox): %s", self._workdir)
            return

        # Non-sandbox (host) path: original behavior — cleanup then remove workdir.
        self._cleanup_previous()
        if os.path.exists(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        logger.debug("LocalOSContainer terminated: %s", self._workdir)

    def execute_independent(
        self, command_item: CommandItem, *parameters: str
    ) -> CommandExecutionResult:
        # L1 guard: never even attempt a command that references a protected path
        # (e.g. /storage). Belt-and-suspenders on top of the ns tmpfs cover.
        if self._sandbox:
            script_text = getattr(command_item, "script", "") or ""
            if _command_references_protected(script_text) or any(
                _command_references_protected(str(p)) for p in parameters
            ):
                logger.warning("Blocked agent command referencing a protected path (/storage).")
                return CommandExecutionResult(
                    exit_code=1,
                    output="Command refused: references a protected path (/storage) which is not accessible in this sandbox.",
                    timeout_flag=False,
                )
        match command_item.command_name:
            case CommandName.BASH:
                cmd = ["bash", "-c", command_item.script]
                if parameters:
                    cmd.append("--")
                    cmd.extend(parameters)
            case CommandName.PYTHON:
                cmd = ["python3", "-c", command_item.script, *parameters]
            case CommandName.CPP:
                src_path = os.path.join(self._workdir, "main.cpp")
                bin_path = os.path.join(self._workdir, "a.out")
                with open(src_path, "w") as f:
                    f.write(command_item.script)
                compile_result = subprocess.run(
                    ["g++", "-o", bin_path, src_path],
                    capture_output=True, text=True, cwd=self._workdir,
                )
                if compile_result.returncode != 0:
                    return CommandExecutionResult(
                        exit_code=compile_result.returncode,
                        output=compile_result.stdout + compile_result.stderr,
                        timeout_flag=False,
                    )
                cmd = [bin_path, *parameters]
            case CommandName.C:
                src_path = os.path.join(self._workdir, "main.c")
                bin_path = os.path.join(self._workdir, "a.out")
                with open(src_path, "w") as f:
                    f.write(command_item.script)
                compile_result = subprocess.run(
                    ["gcc", "-o", bin_path, src_path],
                    capture_output=True, text=True, cwd=self._workdir,
                )
                if compile_result.returncode != 0:
                    return CommandExecutionResult(
                        exit_code=compile_result.returncode,
                        output=compile_result.stdout + compile_result.stderr,
                        timeout_flag=False,
                    )
                cmd = [bin_path, *parameters]
            case _:
                raise NotImplementedError(
                    f"Unsupported command type: {command_item.command_name}"
                )

        return self._execute_with_timeout(cmd)

    def _execute_with_timeout(self, cmd: list) -> CommandExecutionResult:
        # When sandboxed, run the command inside the keeper mount namespace.
        run_cmd = self._wrap_cmd(cmd)

        # Use Popen in its own process group so that on timeout we can KILL the
        # whole tree (nsenter -> bash -> children). CRITICAL: redirect stdout/stderr
        # to a FILE, not a PIPE. If the agent command spawns a background process
        # (e.g. a daemon/`&`/nohup that inherits the stdout fd), Popen.communicate()
        # on a PIPE blocks waiting for pipe EOF FOREVER — even past its timeout and
        # even after killpg — because the surviving grandchild keeps the fd open.
        # With a file + proc.wait(timeout) we only ever wait on the direct child,
        # so the timeout always fires and we can kill the group and read the file.
        import tempfile as _tf
        outf = None
        try:
            outf = _tf.TemporaryFile(mode="w+", dir=self._workdir)
        except Exception:
            outf = _tf.TemporaryFile(mode="w+")

        def _read_outf(limit: int = 200000) -> str:
            try:
                outf.seek(0)
                return outf.read(limit)
            except Exception:
                return ""

        try:
            proc = subprocess.Popen(
                run_cmd,
                stdout=outf,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,   # never block on stdin (e.g. `cat` w/o args)
                cwd="/",
                env={**os.environ, "HOME": "/root"},
                text=True,
                start_new_session=True,  # own process group for group-kill
            )
        except Exception as e:
            try:
                outf.close()
            except Exception:
                pass
            return CommandExecutionResult(exit_code=None, output=f"exec error: {e}", timeout_flag=False)

        try:
            rc = proc.wait(timeout=self.timeout_sec)
            out = _read_outf()
            outf.close()
            return CommandExecutionResult(exit_code=rc, output=out, timeout_flag=False)
        except subprocess.TimeoutExpired:
            # Kill the whole process group (incl. backgrounded grandchildren), reap
            # the direct child (never blocks on pipes since output went to a file).
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(os.getpgid(proc.pid), sig)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=5)
                    break
                except Exception:
                    continue
            out = _read_outf()
            try:
                outf.close()
            except Exception:
                pass
            return CommandExecutionResult(exit_code=None, output=out, timeout_flag=True)

    def __del__(self):
        try:
            self.terminate()
        except Exception:
            pass


# --- Apptainer backend (requires loop devices + SIF image) ---

_active_instances: set = set()


def _cleanup_all_instances():
    for name in list(_active_instances):
        try:
            subprocess.run(
                ["apptainer", "instance", "stop", name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
    _active_instances.clear()


atexit.register(_cleanup_all_instances)

DEFAULT_OS_IMAGE = "/storage/openpsi/images/os-interaction.sif"


class ApptainerOSContainer:
    """Apptainer-based container for OS Interaction tasks."""

    def __init__(
        self,
        command_execution_timeout: int,
        image: str = DEFAULT_OS_IMAGE,
    ):
        self.timeout_sec: float = command_execution_timeout
        self.image = image
        self.instance_name = f"os_{uuid.uuid4().hex[:12]}"
        self._start_instance()

    def _start_instance(self) -> None:
        cmd = [
            "apptainer", "instance", "start",
            "--writable-tmpfs",
            "--no-home",
            "--containall",
            self.image,
            self.instance_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start Apptainer instance '{self.instance_name}': "
                f"{result.stderr.strip()}"
            )
        _active_instances.add(self.instance_name)
        logger.debug("Started Apptainer OS instance: %s", self.instance_name)

    def terminate(self) -> None:
        cmd = ["apptainer", "instance", "stop", self.instance_name]
        subprocess.run(cmd, capture_output=True, timeout=10)
        _active_instances.discard(self.instance_name)
        logger.debug("Stopped Apptainer OS instance: %s", self.instance_name)

    def execute_independent(
        self, command_item: CommandItem, *parameters: str
    ) -> CommandExecutionResult:
        match command_item.command_name:
            case CommandName.BASH:
                cmd = ["bash", "-c", command_item.script]
                if parameters:
                    cmd.append("--")
                    cmd.extend(parameters)
            case CommandName.PYTHON:
                cmd = ["python3", "-c", command_item.script, *parameters]
            case CommandName.CPP:
                compile_script = (
                    f"cat > /tmp/main.cpp << 'CPPEOF'\n"
                    f"{command_item.script}\n"
                    f"CPPEOF\n"
                    f"g++ -o /tmp/a.out /tmp/main.cpp"
                )
                compile_result = self._exec_in_instance(["bash", "-c", compile_script])
                if compile_result.returncode != 0:
                    return CommandExecutionResult(
                        exit_code=compile_result.returncode,
                        output=compile_result.stdout + compile_result.stderr,
                        timeout_flag=False,
                    )
                cmd = ["/tmp/a.out", *parameters]
            case CommandName.C:
                compile_script = (
                    f"cat > /tmp/main.c << 'CEOF'\n"
                    f"{command_item.script}\n"
                    f"CEOF\n"
                    f"gcc -o /tmp/a.out /tmp/main.c"
                )
                compile_result = self._exec_in_instance(["bash", "-c", compile_script])
                if compile_result.returncode != 0:
                    return CommandExecutionResult(
                        exit_code=compile_result.returncode,
                        output=compile_result.stdout + compile_result.stderr,
                        timeout_flag=False,
                    )
                cmd = ["/tmp/a.out", *parameters]
            case _:
                raise NotImplementedError(
                    f"Unsupported command type: {command_item.command_name}"
                )

        return self._execute_with_timeout(cmd)

    def _exec_in_instance(self, cmd: list) -> subprocess.CompletedProcess:
        full_cmd = [
            "apptainer", "exec",
            f"instance://{self.instance_name}",
            *cmd,
        ]
        return subprocess.run(full_cmd, capture_output=True, text=True)

    def _execute_with_timeout(self, cmd: list) -> CommandExecutionResult:
        result_holder: dict = {}

        def run_exec():
            try:
                proc = self._exec_in_instance(cmd)
                result_holder["result"] = CommandExecutionResult(
                    exit_code=proc.returncode,
                    output=(proc.stdout or "") + (proc.stderr or ""),
                    timeout_flag=False,
                )
            except Exception as e:
                result_holder["exception"] = e

        thread = threading.Thread(target=run_exec, daemon=True)
        thread.start()
        thread.join(self.timeout_sec)

        if thread.is_alive():
            return CommandExecutionResult(
                exit_code=None, output=None, timeout_flag=True
            )

        if "exception" in result_holder:
            raise result_holder["exception"]

        return result_holder["result"]
