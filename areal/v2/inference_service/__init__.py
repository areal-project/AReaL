# SPDX-License-Identifier: Apache-2.0

__all__ = ["InferenceServiceWorkflow"]


def __getattr__(name: str):
    # v1 imports the shared Router client without importing the v2 Controller.
    if name == "InferenceServiceWorkflow":
        from areal.v2.inference_service.controller.workflow import (
            InferenceServiceWorkflow,
        )

        return InferenceServiceWorkflow
    raise AttributeError(name)
