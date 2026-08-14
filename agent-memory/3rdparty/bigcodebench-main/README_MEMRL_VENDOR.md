# Vendored BigCodeBench

This directory is a vendored copy of the BigCodeBench reference implementation,
used to run the official evaluation (`bigcodebench.eval.untrusted_check`) in a
fully offline / reproducible way.

Source used during integration:
- `/mnt/public/code/jq/memory_rl/bigcodebench-main`

How it's used in MemRL:
- Our BigCodeBench runner adds `3rdparty/bigcodebench-main` to `sys.path` at runtime
  and imports `bigcodebench.eval` / `bigcodebench.evaluate` from here.

Notes:
- This repo may contain its own tooling/tests; MemRL treats it as a third-party
  dependency and does not run or modify it during normal development.

