# SPDX-License-Identifier: Apache-2.0

"""Explicit closed-form CPU fixture; these seconds are not GPU measurements."""

from dataclasses import asdict

from areal.autopilot.domains.elastic.contracts import OnlineProblem
from areal.autopilot.domains.elastic.evaluation.evaluator import (
    CostModelHistoryBenchmark,
    EvaluatorRevisionBenchmark,
    HistoricalCase,
    HistoricalInput,
    HistorySpan,
)
from areal.autopilot.domains.elastic.evaluation.optimizer import (
    OptimizerRevisionBenchmark,
)
from areal.autopilot.revision.execution import load_source
from areal.elastic.contracts import (
    EpisodeSample,
    Forecast,
    HandoffConfig,
    OnlineLayout,
    SimulationConfig,
    Turn,
)
from areal.elastic.reconfigurator.reference.evaluator import Evaluator
from areal.elastic.serving import ServingSLO
from areal.elastic.simulation.simulation import OnlineSimulation

MODEL_SOURCE = """from areal.elastic.reconfigurator.reference.evaluator.cost_model.contracts import RolloutConfig
class Components:
    source_ids = ("synthetic-online-revision-demo-v1",)
    def __init__(self, scale): self.scale = scale
    def rollout_config(self, layout):
        return RolloutConfig(layout.rollout_replicas,layout.rollout_tp,32,512,1000000)
    def service(self, layout, kv, slices):
        return (0.005+sum(s.query_tokens for s in slices)/100000+kv/10000000)/layout.rollout_tp
    def actor(self, layout, batch):
        return (("logp",0.4/layout.actor_gpus),("advantage",0.2),
                ("train",self.scale*sum(t.input_tokens+t.output_tokens for e in batch for t in e)/100/layout.actor_gpus))
    def publication(self, layout, recipients):
        return 0.001*sum(r.rollout_gpus for r in recipients)
def build_cost_models(calibration_cases, settings):
    return Components(SCALE)
"""
GOOD_EVALUATOR_SOURCE = MODEL_SOURCE.replace("SCALE", "1.0")
BIASED_EVALUATOR_SOURCE = MODEL_SOURCE.replace("SCALE", "1.3")
REDUNDANT_OPTIMIZER_SOURCE = """from areal.elastic.reconfigurator.reference.optimizer import OnlineOptimizer
class Redundant(OnlineOptimizer):
    def search(self, evaluator, view, forecasts, layouts, **kwargs):
        self.evaluate_many(evaluator,view,forecasts,layouts)
        return self.evaluate_many(evaluator,view,forecasts,layouts)
def build_optimizer(settings): return Redundant()
"""


def demo_inputs():
    initial = OnlineLayout(2, 1, 2, 1)
    candidates = (initial, OnlineLayout(1, 1, 1, 1), OnlineLayout(1, 1, 2, 1))
    run = SimulationConfig(4, 2, 32, (("python", 8),), 4096, 65536, 1000000, 1000)
    handoff = HandoffConfig(
        (8, 8, 8, 8),
        0,
        0,
        0,
        0,
        1000,
        1000,
        1,
        0,
        0,
        "Synthetic CPU-only zero startup / explicit transfer prior",
    )
    slo = ServingSLO(5, 0.1, 0.8)
    cost_models = load_source(GOOD_EVALUATOR_SOURCE).build_cost_models((), {})
    workload = tuple(
        EpisodeSample(f"demo-{i}", "math", i, (Turn(16, 4),)) for i in range(8)
    )
    simulation = OnlineSimulation(
        workload, initial, cost_models, run, handoff, slo, input_closes_at=20
    )
    problem = OnlineProblem(
        simulation.observe(), (Forecast(0, workload, 20),), candidates
    )
    cases = []
    for i in range(10):
        layout = candidates[i % len(candidates)]
        tokens = 20 + i * 7
        batch = tuple((Turn(tokens, 4),) for _ in range(4))
        measured = (
            0.4 / layout.actor_gpus + 0.2 + 4 * (tokens + 4) / 100 / layout.actor_gpus
        )
        inputs = HistoricalInput(
            "actor",
            layout,
            "synthetic-formula-v1",
            "full-logp-advantage-train",
            batch=batch,
        )
        cases.append(
            HistoricalCase(
                str(i),
                HistorySpan("demo", i, i),
                "calibration" if i < 2 else "final_holdout" if i == 9 else "validation",
                inputs,
                (("actor_total_seconds", measured),),
                "independent-closed-form-fixture",
            )
        )
    assembly = dict(
        run=asdict(run),
        handoff=asdict(handoff),
        slo=asdict(slo),
        candidates=[asdict(c) for c in candidates],
        reconfigurator={},
        forecast=dict(
            history_seconds=120,
            horizon_seconds=20,
            samples=1,
            minimum_completed_episodes=1,
            seed=0,
            max_new_tokens=32,
            unseen_tool_residual_seconds=1,
        ),
    )
    return cases, (problem,), assembly


def demo_benchmark(target, runner=None):
    cases, problems, assembly = demo_inputs()
    if target == "evaluator":
        history = CostModelHistoryBenchmark(
            tuple(cases), primary_metrics=("actor_total_seconds",)
        )
        return EvaluatorRevisionBenchmark(
            history,
            settings={"fixture": "synthetic-online"},
            runner=runner,
            initial_source=BIASED_EVALUATOR_SOURCE,
        )
    cost_models = load_source(GOOD_EVALUATOR_SOURCE).build_cost_models((), {})
    from areal.autopilot.domains.elastic.contracts import decode

    return OptimizerRevisionBenchmark(
        problems,
        evaluator_factory=lambda: Evaluator(
            cost_models,
            decode(SimulationConfig, assembly["run"]),
            decode(HandoffConfig, assembly["handoff"]),
            decode(ServingSLO, assembly["slo"]),
        ),
        context={"fixture": "synthetic-online"},
        runner=runner,
        max_queries=12,
        overhead_weight=0.5,
    )
