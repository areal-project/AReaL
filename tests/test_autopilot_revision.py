# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


def test_revision_core_runs_an_unrelated_domain_without_elastic_imports(tmp_path):
    """A domain port can accept and resume revisions with Elastic unavailable."""
    import subprocess
    import textwrap

    source = textwrap.dedent("""
        import sys, types, importlib.abc
        from dataclasses import dataclass
        from pathlib import Path
        root, store = map(Path, sys.argv[1:])
        package = types.ModuleType("areal")
        package.__path__ = [str(root / "areal")]
        sys.modules["areal"] = package
        class BlockDomain(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.startswith(("areal.elastic", "areal.autopilot.domains")):
                    raise AssertionError("Unexpected domain dependency: " + fullname)
        sys.meta_path.insert(0, BlockDomain())
        from areal.autopilot.revision import RevisionLoop, Proposal, LLMRevisionAgent
        from areal.autopilot.revision.contracts import EvaluationReport, source_digest
        @dataclass(frozen=True)
        class Criteria:
            def compare(self, candidate, incumbent):
                return candidate.score < incumbent.score, "measured error"
        class Evaluator:
            target = "text-normalizer"
            benchmark_sha256 = "fixed-text-corpus"
            initial_source = "error = 2"
            owns_acceptance = False
            criteria = Criteria()
            context = settings = calibration = {}
            proposal_context = {"api": "normalize(text) -> str", "reference_sources": {}}
            def evaluate(self, source, directory):
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "candidate.py").write_text(source)
                return EvaluationReport(self.target, source_digest(source), self.benchmark_sha256,
                                        True, int(source[-1]), {})
        class Agent:
            calls = 0
            def propose(self, request):
                self.calls += 1
                assert request.to_dict()["api"] == "normalize(text) -> str"
                assert "reference_sources" in LLMRevisionAgent("test").messages(request)[1]["content"]
                return Proposal("error = 1")
        agent, evaluator = Agent(), Evaluator()
        result = RevisionLoop(agent, evaluator, store).run(max_rounds=1)
        assert result["accepted"] == 1 and agent.calls == 1
        result = RevisionLoop(agent, evaluator, store).run(max_rounds=1)
        assert result["accepted"] == 0 and result["final_score"] == 1
        assert not any(n.startswith(("areal.elastic", "areal.autopilot.domains")) for n in sys.modules)
    """)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            source,
            str(Path(__file__).resolve().parents[1]),
            str(tmp_path / "revision"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_elastic_planning_and_native_controls_import_without_autopilot():
    """The AReaL capability remains usable without the AutoPilot extension."""
    import subprocess
    import textwrap

    source = textwrap.dedent("""
        import sys, types, importlib.abc
        from pathlib import Path
        p = types.ModuleType("areal")
        p.__path__ = [str(Path(sys.argv[1]) / "areal")]
        sys.modules["areal"] = p
        class BlockAutoPilot(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, *args):
                if fullname.startswith("areal.autopilot"):
                    raise AssertionError(fullname)
        sys.meta_path.insert(0, BlockAutoPilot())
        from areal.elastic.reconfigurator import basic as planning, reference
        from areal.elastic import contracts, layout
        from areal.elastic.infra import adapter, activation, reconfiguration
        from areal.elastic.infra.state import checkpoint_store
        from areal.elastic.infra.monitor import collector
        assert planning.OnlineReconfigurator and reference.OnlineReconfigurator and reference.Catalog
        assert not any(n.startswith("areal.autopilot") for n in sys.modules)
    """)
    result = subprocess.run(
        [sys.executable, "-c", source, str(Path(__file__).resolve().parents[1])],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
