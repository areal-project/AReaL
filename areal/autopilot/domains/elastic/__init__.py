# SPDX-License-Identifier: Apache-2.0

"""Online Reference revision; generic revision machinery lives outside this domain."""

from areal.autopilot.domains.elastic.cycle import (
    ReconfiguratorRevisionCycle,
    optimizer_simulation_cases,
    revise_optimizer,
)
from areal.autopilot.domains.elastic.evaluation.evaluator import (
    CostModelAcceptanceCriteria,
    CostModelHistoryBenchmark,
    EvaluatorRevisionBenchmark,
    HistoricalCase,
    HistoricalInput,
    HistorySpan,
)
from areal.autopilot.domains.elastic.evaluation.optimizer import (
    OptimizerRevisionBenchmark,
    OptimizerRevisionCriteria,
    OptimizerSimulationBenchmark,
    OptimizerSimulationCriteria,
)
from areal.autopilot.domains.elastic.trigger import (
    RevisionTrigger,
    RevisionTriggerConfig,
)
from areal.autopilot.domains.elastic.versions import (
    load_accepted_cost_models,
    load_accepted_optimizer,
)

__all__ = [
    "ReconfiguratorRevisionCycle",
    "RevisionTrigger",
    "RevisionTriggerConfig",
    "optimizer_simulation_cases",
    "revise_optimizer",
    "load_accepted_cost_models",
    "load_accepted_optimizer",
    "HistoricalCase",
    "HistoricalInput",
    "CostModelHistoryBenchmark",
    "HistorySpan",
    "CostModelAcceptanceCriteria",
    "EvaluatorRevisionBenchmark",
    "OptimizerRevisionCriteria",
    "OptimizerRevisionBenchmark",
    "OptimizerSimulationBenchmark",
    "OptimizerSimulationCriteria",
]
