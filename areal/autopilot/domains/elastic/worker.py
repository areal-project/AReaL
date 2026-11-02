# SPDX-License-Identifier: Apache-2.0

"""Online revision subprocess: public inputs only, bounded by CandidateRunner."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from areal.autopilot.revision.execution import load_module
from areal.elastic.contracts import CandidateWave
from areal.elastic.serving import ServingSLO

from .contracts import OnlineProblem, decode
from .model_adapters import OnlineHistoryPredictor, validate_components
from .model_execution import (
    historical_case_from_dict,
    historical_input_from_dict,
    prediction_fingerprint,
)


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

    class ReferenceProxy:
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
        if request["target"] == "model":
            models = validate_components(
                module.build_model(
                    tuple(
                        historical_case_from_dict(c)
                        for c in request["calibration"]["cases"]
                    ),
                    request["settings"],
                )
            )
            predictor = OnlineHistoryPredictor(models)
            cases = tuple(historical_input_from_dict(c) for c in request["cases"])
            rows = [{"prediction": asdict(predictor.predict(c))} for c in cases]
            repeated = [{"prediction": asdict(predictor.predict(c))} for c in cases]
            if prediction_fingerprint(rows) != prediction_fingerprint(repeated):
                raise ValueError("Frozen Online components must predict repeatably")
            send(dict(type="result", rows=rows, source_ids=tuple(models.source_ids)))
        elif request["target"] == "optimizer":
            predictions = []
            for index, data in enumerate(request["cases"]):
                case = decode(OnlineProblem, data)
                proxy = ReferenceProxy(index, case)
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
            raise ValueError("Online revision target must be model or optimizer")
    except Exception as error:
        send(dict(type="result", error=f"{type(error).__name__}: {error}"))


if __name__ == "__main__":
    main()
