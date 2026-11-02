# SPDX-License-Identifier: Apache-2.0

"""Run Online Reference model/optimizer revisions or a synthetic CPU demonstration."""

import argparse
import json
import sys
from pathlib import Path

from examples.autopilot.elastic.revision_demo import (
    GOOD_MODEL_SOURCE,
    REDUNDANT_OPTIMIZER_SOURCE,
    demo_evaluator,
)

from areal.autopilot.domains.elastic.contracts import BASELINE_SOURCE
from areal.autopilot.domains.elastic.model_evaluator import HistoricalModelEvaluator
from areal.autopilot.domains.elastic.model_execution import (
    HistoricalSourceEvaluator,
    historical_case_from_dict,
)
from areal.autopilot.domains.elastic.optimizer_evaluator import OnlineOptimizerEvaluator
from areal.autopilot.revision import CommandRevisionAgent, RevisionLoop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["model", "optimizer"])
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
            GOOD_MODEL_SOURCE
            if request["target"] == "model"
            else BASELINE_SOURCE["optimizer"]
        )
        sys.stdout.write(
            json.dumps(
                {
                    "source": source,
                    "rationale": "Controlled CPU example: improve model timing or remove redundant model queries",
                }
            )
        )
        return
    if args.target is None or args.output is None:
        parser.error("--target and --output are required")
    if args.demo:
        evaluator = demo_evaluator(args.target)
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
        if args.target == "model":
            history = HistoricalModelEvaluator(
                tuple(historical_case_from_dict(c) for c in cases),
                primary_metrics=tuple(profile["primary_metrics"]),
            )
            evaluator = HistoricalSourceEvaluator(
                history,
                settings=profile["model_settings"],
                initial_source=profile.get("initial_model_source"),
            )
        else:
            from areal.autopilot.domains.elastic.contracts import decode
            from areal.autopilot.revision.execution import load_source
            from areal.elastic.reconfigurator.reference.evaluator import Evaluator
            from areal.elastic.serving import ServingSLO
            from areal.elastic.simulation.online_transitions import HandoffConfig
            from areal.elastic.simulation.simulation_core import SimulationConfig

            assembly = profile["assembly"]
            models = load_source(profile["model_source"]).build_model(
                tuple(
                    historical_case_from_dict(c)
                    for c in profile.get("calibration_cases", ())
                ),
                profile["model_settings"],
            )
            evaluator = OnlineOptimizerEvaluator(
                cases,
                model_factory=lambda: Evaluator(
                    models,
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
    loop = RevisionLoop(agent, evaluator, args.output, initial_source=initial_source)
    result = loop.run(max_rounds=args.max_rounds)
    if (
        args.demo
        and not (args.agent_command or args.llm_model)
        and result["active_source"] is None
    ):
        raise RuntimeError("Synthetic demo did not accept the expected improvement")


if __name__ == "__main__":
    main()
