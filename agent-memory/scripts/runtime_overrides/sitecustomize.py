"""Load job-local MemRL overrides without modifying root-owned source files."""
from pathlib import Path
_override_root = Path(__file__).resolve().parent
try:
    import memrl
    import memrl.apptainer
    import memrl.service
    p1 = str(_override_root / "memrl")
    p2 = str(_override_root / "memrl" / "apptainer")
    p3 = str(_override_root / "memrl" / "service")
    if p1 not in memrl.__path__:
        memrl.__path__.insert(0, p1)
    if p2 not in memrl.apptainer.__path__:
        memrl.apptainer.__path__.insert(0, p2)
    if p3 not in memrl.service.__path__:
        memrl.service.__path__.insert(0, p3)
except Exception:
    pass
