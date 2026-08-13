# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import flask

from areal.v2.training_service.worker.awex import create_awex_blueprint
from areal.v2.weight_update.awex.megatron_adapter import AwexMegatronAdapter


class _MetaServer:
    def __init__(self, events):
        self.events = events

    def wait_set_until_size(self, key, size, timeout):
        self.events.append(("wait", key, size, timeout))

    def delete_if_exists(self, key):
        self.events.append(("delete", key))


def _adapter(events):
    adapter = AwexMegatronAdapter.__new__(AwexMegatronAdapter)
    adapter._engine = SimpleNamespace(cpu_group="cpu")
    adapter._meta_server_client = _MetaServer(events)
    adapter._num_infer_engines = 4
    adapter._timeout_s = 120.0
    return adapter


def test_finish_colocate_update_cleans_reusable_keys_between_barriers():
    events = []
    adapter = _adapter(events)

    with (
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
            return_value=8,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_rank",
            return_value=0,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.barrier",
            side_effect=lambda group: events.append(("barrier", group)),
        ),
    ):
        adapter.finish_colocate_weight_update(training_world_size=8)

    assert events == [
        ("barrier", "cpu"),
        ("wait", "finished_weights_update_engines", 4, 120.0),
        ("delete", "finished_weights_update_engines"),
        ("delete", "all_training_offloaded_weights"),
        ("barrier", "cpu"),
    ]


def test_nonzero_rank_only_participates_in_finish_barriers():
    events = []
    adapter = _adapter(events)

    with (
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_world_size",
            return_value=8,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.get_rank",
            return_value=3,
        ),
        mock.patch(
            "areal.v2.weight_update.awex.megatron_adapter.dist.barrier",
            side_effect=lambda group: events.append(("barrier", group)),
        ),
    ):
        adapter.finish_colocate_weight_update(training_world_size=8)

    assert events == [("barrier", "cpu"), ("barrier", "cpu")]


def test_worker_execute_endpoint_runs_finish_after_transfer():
    events = []
    adapter = SimpleNamespace(
        parallelism_strategy={"world_size": 8},
        enable_colocate_memory_management=lambda: events.append("enable"),
        init_colocate_weight_update=lambda **_kwargs: events.append("init"),
        execute_colocate_weight_update=lambda version: events.append(
            ("execute", version)
        ),
        finish_colocate_weight_update=lambda training_world_size: events.append(
            ("finish", training_world_size)
        ),
        release_memory=lambda tags=None: None,
        resume_memory=lambda tags=None: None,
        teardown_colocate_weight_update=lambda: None,
    )

    def submit_to_engine_thread(_name, action):
        return action()

    def run_endpoint(_name, action, return_result=True):
        result = action()
        return flask.jsonify({"result": result} if return_result else {"status": "ok"})

    with mock.patch(
        "areal.v2.training_service.worker.awex._create_training_adapter",
        return_value=adapter,
    ):
        app = flask.Flask(__name__)
        app.register_blueprint(
            create_awex_blueprint(
                flask_module=flask,
                get_engine=lambda: object(),
                submit_to_engine_thread=submit_to_engine_thread,
                run_endpoint=run_endpoint,
            )
        )
        response = app.test_client().post(
            "/awex/execute_colocate_weight_update", json={"version": 3}
        )

    assert response.status_code == 200
    assert events == [("execute", 3), ("finish", 8)]
