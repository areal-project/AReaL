"""Package marker for the inference-service tests.

This previously registered stub ``areal`` namespace modules on Python < 3.12,
because the top-level ``areal/__init__.py`` used PEP 695 syntax that 3.11
cannot parse. That syntax is gone, so the real package imports cleanly on
every supported interpreter. The stubs are worse than unnecessary now: by
skipping ``areal/__init__.py`` they let ``areal.api.cli_args`` become the entry
point of the import graph, which exposes an ``api -> engine -> infra -> api``
cycle that the real ``__init__`` masks by importing ``areal.infra`` first.
"""
