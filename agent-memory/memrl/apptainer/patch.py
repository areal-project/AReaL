"""Monkey-patch LLB container modules to use local/Apptainer implementations.

Call `patch_llb_containers()` before any LLB task code imports the container
classes. This replaces Docker-based containers with non-Docker equivalents.

Backend selection:
- OS tasks: defaults to LocalOSContainer (subprocess-based, no container needed).
  Set MEMRL_OS_BACKEND=apptainer to use ApptainerOSContainer.
- DB tasks: defaults to LocalDBContainer if mysqld is on PATH,
  otherwise ApptainerDBContainer if Apptainer is available.
  Set MEMRL_DB_BACKEND=local|apptainer to override.
"""

import logging
import os
import shutil

logger = logging.getLogger(__name__)

_patched = False


def _has_apptainer() -> bool:
    return shutil.which("apptainer") is not None or shutil.which("singularity") is not None


def _has_mysqld() -> bool:
    if shutil.which("mysqld") is not None or shutil.which("mariadbd") is not None:
        return True
    for candidate in ["/usr/libexec/mysqld", "/usr/sbin/mysqld", "/usr/sbin/mariadbd"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return True
    return False


def patch_llb_containers() -> None:
    """Replace LLB Docker containers with local/Apptainer implementations.

    This is a no-op if MEMRL_CONTAINER_BACKEND=docker is set.
    """
    global _patched
    if _patched:
        return

    backend = os.environ.get("MEMRL_CONTAINER_BACKEND", "").lower()
    if backend == "docker":
        logger.info("Container backend forced to Docker via MEMRL_CONTAINER_BACKEND")
        return

    # --- OS container ---
    os_backend = os.environ.get("MEMRL_OS_BACKEND", "local").lower()
    if os_backend == "apptainer" and _has_apptainer():
        from memrl.apptainer.os_container import ApptainerOSContainer as OSClass
        logger.info("OS container backend: Apptainer")
    else:
        from memrl.apptainer.os_container import LocalOSContainer as OSClass
        logger.info("OS container backend: Local (subprocess)")

    import src.tasks.instance.os_interaction.container as os_mod
    os_mod.OSInteractionContainer = OSClass  # type: ignore[attr-defined]

    # Also patch the task module's reference (it may have already imported the class)
    try:
        import src.tasks.instance.os_interaction.task as os_task_mod
        os_task_mod.OSInteractionContainer = OSClass  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    # --- DB container ---
    db_backend = os.environ.get("MEMRL_DB_BACKEND", "auto").lower()
    if db_backend == "apptainer" or (db_backend == "auto" and not _has_mysqld() and _has_apptainer()):
        from memrl.apptainer.db_container import ApptainerDBContainer as DBClass
        logger.info("DB container backend: Apptainer")
    elif db_backend == "local" or (db_backend == "auto" and _has_mysqld()):
        from memrl.apptainer.db_container import LocalDBContainer as DBClass
        logger.info("DB container backend: Local (mysqld)")
    else:
        logger.warning(
            "No suitable DB backend found (no mysqld, no apptainer). "
            "DB Bench tasks will fail. Install mysqld or provide Apptainer + MySQL SIF."
        )
        _patched = True
        return

    import src.tasks.instance.db_bench.container as db_mod
    db_mod.DBBenchContainer = DBClass  # type: ignore[attr-defined]

    # Also patch the task module's reference
    try:
        import src.tasks.instance.db_bench.task as db_task_mod
        db_task_mod.DBBenchContainer = DBClass  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    _patched = True
    logger.info("LLB container patching complete")
