<!-- Template for per-package API checklists. Copy this file and fill in for each package. -->

______________________________________________________________________

package: <pip-package-name> github: \<org/repo> branch_template: "v${VERSION}"
upstream_paths:

- path/to/relevant/file.py
- path/to/relevant/module/

______________________________________________________________________

## Affected Files

### Primary (engine layer — most likely to break)

| File                    | Imports / Usage                   |
| ----------------------- | --------------------------------- |
| `areal/path/to/file.py` | `module.function`, `module.Class` |

### Secondary (model / infra layer)

| File                    | Imports / Usage   |
| ----------------------- | ----------------- |
| `areal/path/to/file.py` | `module.function` |

### Tertiary (tests, config)

| File                 | Imports / Usage   |
| -------------------- | ----------------- |
| `tests/test_file.py` | `module.function` |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, **changed method
signatures on returned objects**, and **moved/renamed modules**.

### 1. `module.submodule.FunctionOrClass`

**Source:** `upstream-repo/path/to/file.py`

Called in `areal/path/to/file.py`:

```python
# Paste the actual call site code here
FunctionOrClass(param1=..., param2=...)
```

**Check:** \[Describe what to verify — new params? renamed? return type change?\]

### 2. ...

______________________________________________________________________

## Version-Guarded Code

<!-- List any AReaL code that has version-specific behavior for this package -->

- `areal/path/to/file.py:LINE` — description of version guard and threshold
