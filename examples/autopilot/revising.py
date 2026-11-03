# SPDX-License-Identifier: Apache-2.0

"""Run Reference Evaluator/Optimizer revisions or a synthetic CPU demonstration."""

import argparse
import json
import sys
from pathlib import Path

from examples.autopilot.elastic.revision_demo import (
    GOOD_EVALUATOR_SOURCE,
    REDUNDANT_OPTIMIZER_SOURCE,
    demo_benchmark,
)

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
from areal.autopilot.domains.elastic.evaluation.evaluator import (
    CostModelHistoryBenchmark,
    EvaluatorRevisionBenchmark,
    historical_case_from_dict,
)
from areal.autopilot.domains.elastic.evaluation.optimizer import (
    OptimizerRevisionBenchmark,
)
from areal.autopilot.revision import CommandRevisionAgent, RevisionLoop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["evaluator", "optimizer"])
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--demo", action="store_true")
    proposer = parser.add_mutually_exclusive_group()
    proposer.add_argument("--demo-agent", action="store_true")
    proposer.add_argument("--agent-command", nargs=argparse.REMAINDER)
    proposer.add_argument("--llm-model")
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-timeout", type=float, default=180)
    parser.add_argument("--llm-max-output-tokens", type=int, default=8192)
    parser.add_argument(
        "--llm-response-format",
        choices=["json_schema", "json_object"],
        default="json_schema",
    )
    parser.add_argument("--llm-reasoning-effort")
    parser.add_argument(
        "--llm-token-parameter",
        choices=["max_completion_tokens", "max_tokens"],
        default="max_completion_tokens",
    )
    args = parser.parse_args()
    if args.demo_agent:
        request = json.load(sys.stdin)
        source = (
            GOOD_EVALUATOR_SOURCE
            if request["target"] == "evaluator"
            else BASELINE_SOURCE["optimizer"]
        )
        sys.stdout.write(
            json.dumps(
                {
                    "source": source,
                    "rationale": "Controlled CPU example: improve CostModel timing or remove redundant Evaluator queries",
                }
            )
        )
        return
    if args.target is None or args.output is None:
        parser.error("--target and --output are required")
    if args.demo:
        benchmark = demo_benchmark(args.target)
    else:
        if (
            args.profile is None
            or args.cases is None
            or not (args.agent_command or args.llm_model)
        ):
            parser.error(
                "Supply --profile, --cases and either --llm-model or --agent-command; "
                "or use --demo"
            )
        profile, cases = (
            json.loads(args.profile.read_text()),
            json.loads(args.cases.read_text()),
        )
    if args.llm_model:
        command = [
            sys.executable,
            "-m",
            "areal.autopilot.revision",
            "--model",
            args.llm_model,
            "--api-key-env",
            args.llm_api_key_env,
            "--timeout",
            str(args.llm_timeout),
            "--max-output-tokens",
            str(args.llm_max_output_tokens),
            "--response-format",
            args.llm_response_format,
            "--token-parameter",
            args.llm_token_parameter,
        ]
        if args.llm_base_url:
            command.extend(["--base-url", args.llm_base_url])
        if args.llm_reasoning_effort:
            command.extend(["--reasoning-effort", args.llm_reasoning_effort])
        agent = CommandRevisionAgent(tuple(command), args.llm_timeout + 10)
    else:
        command = (
            tuple(args.agent_command)
            if args.agent_command
            else (sys.executable, "-m", "examples.autopilot.revising", "--demo-agent")
        )
        agent = CommandRevisionAgent(command)
    if not args.demo:
        settings = (
            {} if args.settings is None else json.loads(args.settings.read_text())
        )
        if args.target == "evaluator":
            history = CostModelHistoryBenchmark(
                tuple(historical_case_from_dict(c) for c in cases),
                primary_metrics=tuple(profile["primary_metrics"]),
            )
            benchmark = EvaluatorRevisionBenchmark(
                history,
                settings=profile.get(
                    "cost_model_settings", profile.get("model_settings")
                ),
                initial_source=profile.get(
                    "initial_evaluator_source", profile.get("initial_model_source")
                ),
            )
        else:
            from areal.autopilot.domains.elastic.contracts import decode
            from areal.autopilot.revision.execution import load_source
            from areal.elastic.contracts import HandoffConfig, SimulationConfig
            from areal.elastic.reconfigurator.reference.evaluator import Evaluator
            from areal.elastic.serving import ServingSLO

            assembly = profile["assembly"]
            cost_models = load_source(
                profile.get("evaluator_source", profile.get("model_source"))
            ).build_cost_models(
                tuple(
                    historical_case_from_dict(c)
                    for c in profile.get("calibration_cases", ())
                ),
                profile.get("cost_model_settings", profile.get("model_settings")),
            )
            benchmark = OptimizerRevisionBenchmark(
                cases,
                evaluator_factory=lambda: Evaluator(
                    cost_models,
                    decode(SimulationConfig, assembly["run"]),
                    decode(HandoffConfig, assembly["handoff"]),
                    decode(ServingSLO, assembly["slo"]),
                ),
                context={"assembly": assembly},
                settings=settings,
            )
    initial_source = (
        REDUNDANT_OPTIMIZER_SOURCE if args.demo and args.target == "optimizer" else None
    )
    loop = RevisionLoop(agent, benchmark, args.output, initial_source=initial_source)
    result = loop.run(max_rounds=args.max_rounds)
    if (
        args.demo
        and not (args.agent_command or args.llm_model)
        and result["active_source"] is None
    ):
        raise RuntimeError("Synthetic demo did not accept the expected improvement")


if __name__ == "__main__":
    main()
