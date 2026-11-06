# SPDX-License-Identifier: Apache-2.0

"""Candidate execution worker for Elastic revision evaluation."""

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from areal.autopilot.domains.elastic.contracts import OnlineProblem, decode
from areal.autopilot.domains.elastic.evaluation.evaluator import (
    CostModelHistoryPredictor,
    historical_case_from_dict,
    historical_input_from_dict,
    prediction_fingerprint,
    validate_cost_models,
)
from areal.autopilot.revision.execution import load_module
from areal.elastic.contracts import CandidateWave
from areal.elastic.serving import ServingSLO


def run_optimizer_search(source, ports, settings, runner, directory):
    """Run frozen searches in the child, serving their Evaluator calls here."""
    from areal.elastic.contracts import CandidatePrediction, OnlineLayout
    from areal.elastic.reconfigurator.reference.optimizer import SearchBound

    begun, finished = set(), {}

    def simulate(message):
        index = message["case"]
        if type(index) is not int or not 0 <= index < len(ports):
            raise ValueError("Unknown Online case")
        port, case = ports[index], ports[index].problem
        operation = message["operation"]
        if operation == "start":
            if index in begun:
                return port.reject("Search case started twice")
            begun.add(index)
            port.started = time.monotonic()
            return dict(
                context_id=port.context_id,
                historical=port.historical,
                slo=asdict(port.slo),
                parallel_available=port.parallel_available,
            )
        if index not in begun:
            return port.reject("Search queried before starting its case")
        if operation == "order":
            return {
                "order": [
                    (i, asdict(layout))
                    for i, layout in port.order_candidates(
                        case.observation,
                        case.forecasts,
                        case.candidates,
                        case.preferred,
                    )
                ]
            }
        if operation == "wave":
            raw = message.get("bound")
            bound = (
                None
                if raw is None
                else SearchBound(
                    port.context_id,
                    raw["target"],
                    decode(CandidatePrediction, raw["best"]),
                    raw["fraction_pruning"],
                )
            )
            wave = port.evaluate_wave(
                case.observation,
                case.forecasts,
                tuple(
                    (i, decode(OnlineLayout, layout))
                    for i, layout in message["indexed"]
                ),
                bound=bound,
                chunk_size=message.get("chunk_size", 2),
            )
            return {"wave": asdict(wave)}
        if operation == "finish":
            finished[index] = port.finish(
                tuple(
                    decode(CandidatePrediction, value)
                    for value in message["predictions"]
                )
            )
            return {"finished": True}
        return port.reject("Unknown configuration Evaluator operation")

    output = runner.run(
        source,
        dict(
            target="optimizer",
            settings=settings,
            cases=[asdict(port.problem) for port in ports],
        ),
        directory,
        simulate,
    )
    if output.get("error"):
        raise ValueError(output["error"])
    if len(output["predictions"]) != len(ports):
        raise ValueError("Search omitted cases")
    predictions = []
    for index, (port, raw) in enumerate(zip(ports, output["predictions"], strict=True)):
        result = tuple(decode(CandidatePrediction, value) for value in raw)
        if index not in finished or result != finished[index] or port.violations:
            raise ValueError(
                "Search result lacks its completed parent-owned evaluation"
            )
        predictions.append(result)
    return tuple(predictions)


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    incoming = os.fdopen(int(sys.argv[2]), "r")
    outgoing = os.fdopen(int(sys.argv[3]), "w", buffering=1)

    def send(value):
        outgoing.write(json.dumps(value, allow_nan=False) + "\n")

    def rpc(index, operation, **data):
        send(dict(type="simulate", case=index, operation=operation, **data))
        response = json.loads(incoming.readline())
        if "error" in response:
            raise ValueError(response["error"])
        return response

    class EvaluatorProxy:
        def __init__(self, index, case):
            self.index, self.case = index, case
            metadata = rpc(index, "start")
            self.context_id = metadata["context_id"]
            self.historical = metadata["historical"]
            self.parallel_available = metadata["parallel_available"]
            self.slo = decode(ServingSLO, metadata["slo"])
            self.execution_stats = {}

        def prepare_context(self, view, forecasts):
            if view != self.case.observation or forecasts != self.case.forecasts:
                raise ValueError("Search changed its public Online inputs")

        def order_candidates(self, view, forecasts, layouts, preferred=()):
            self.prepare_context(view, forecasts)
            if (
                tuple(layouts) != self.case.candidates
                or tuple(preferred) != self.case.preferred
            ):
                raise ValueError("Search changed its catalog or preferred layouts")
            from areal.elastic.contracts import OnlineLayout

            return tuple(
                (i, decode(OnlineLayout, c))
                for i, c in rpc(self.index, "order")["order"]
            )

        def evaluate_wave(self, view, forecasts, indexed, *, bound=None, chunk_size=2):
            self.prepare_context(view, forecasts)
            value = rpc(
                self.index,
                "wave",
                indexed=[(i, asdict(c)) for i, c in indexed],
                bound=None if bound is None else asdict(bound),
                chunk_size=chunk_size,
            )
            return decode(CandidateWave, value["wave"])

        def evaluate(self, view, forecasts, layout, *, bound=None):
            return self.evaluate_wave(
                view, forecasts, ((0, layout),), bound=bound
            ).predictions[0]

        def evaluate_many(self, view, forecasts, layouts):
            return self.evaluate_wave(
                view, forecasts, tuple(enumerate(layouts)), chunk_size=8
            ).predictions

    try:
        module = load_module(Path(request["source_path"]))
        if request["target"] == "evaluator":
            cost_models = validate_cost_models(
                module.build_cost_models(
                    tuple(
                        historical_case_from_dict(c)
                        for c in request["calibration"]["cases"]
                    ),
                    request["settings"],
                )
            )
            predictor = CostModelHistoryPredictor(cost_models)
            cases = tuple(historical_input_from_dict(c) for c in request["cases"])
            rows = [{"prediction": asdict(predictor.predict(c))} for c in cases]
            repeated = [{"prediction": asdict(predictor.predict(c))} for c in cases]
            if prediction_fingerprint(rows) != prediction_fingerprint(repeated):
                raise ValueError("Frozen Online components must predict repeatably")
            send(
                dict(type="result", rows=rows, source_ids=tuple(cost_models.source_ids))
            )
        elif request["target"] == "optimizer":
            predictions = []
            for index, data in enumerate(request["cases"]):
                case = decode(OnlineProblem, data)
                proxy = EvaluatorProxy(index, case)
                search = module.build_optimizer(dict(request["settings"]))
                result = tuple(
                    search.search(
                        proxy,
                        case.observation,
                        case.forecasts,
                        case.candidates,
                        preferred=case.preferred,
                        trigger=case.trigger,
                    )
                )
                raw = [asdict(p) for p in result]
                rpc(index, "finish", predictions=raw)
                predictions.append(raw)
            send(dict(type="result", predictions=predictions))
        else:
            raise ValueError("Online revision target must be evaluator or optimizer")
    except Exception as error:
        send(dict(type="result", error=f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    main()
