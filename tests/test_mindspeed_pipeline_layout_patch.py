# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

from areal.engine.megatron_utils.mindspeed_pipeline_layout_patch import (
    ensure_mindspeed_pipeline_layout_stage_count,
)


def _install_fake_fb_overlap(monkeypatch):
    for package_name in (
        "mindspeed",
        "mindspeed.features_manager",
        "mindspeed.features_manager.moe",
    ):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    class MoEFwdBwdOverlapFeature:
        @staticmethod
        def _get_pipeline_model_parallel_layout_stage_count(layout):
            return len(layout.replace(",", "").split("|"))

        @classmethod
        def _has_virtual_pipeline(cls, args):
            layout = args.pipeline_model_parallel_layout
            num_stages = cls._get_pipeline_model_parallel_layout_stage_count(layout)
            return num_stages // args.pipeline_model_parallel_size > 1

        def validate_args(self, args):
            incorrect_schedule = (
                args.schedules_method != "dualpipev"
                and not self._has_virtual_pipeline(args)
                and args.pipeline_model_parallel_size != 1
            )
            if args.moe_fb_overlap and incorrect_schedule:
                raise AssertionError

    module_name = "mindspeed.features_manager.moe.fb_overlap"
    module = ModuleType(module_name)
    module.MoEFwdBwdOverlapFeature = MoEFwdBwdOverlapFeature
    monkeypatch.setitem(sys.modules, module_name, module)
    return MoEFwdBwdOverlapFeature


def test_stage_count_supports_mcore_layout_objects(monkeypatch):
    feature = _install_fake_fb_overlap(monkeypatch)
    layout = SimpleNamespace(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=3,
    )

    assert ensure_mindspeed_pipeline_layout_stage_count()
    assert feature._get_pipeline_model_parallel_layout_stage_count(layout) == 6

    args = SimpleNamespace(
        moe_fb_overlap=False,
        pipeline_model_parallel_layout=layout,
        pipeline_model_parallel_size=2,
        schedules_method=None,
    )
    feature().validate_args(args)


def test_stage_count_patch_is_idempotent(monkeypatch):
    feature = _install_fake_fb_overlap(monkeypatch)

    assert ensure_mindspeed_pipeline_layout_stage_count()
    wrapped = feature._get_pipeline_model_parallel_layout_stage_count
    assert ensure_mindspeed_pipeline_layout_stage_count()

    assert feature._get_pipeline_model_parallel_layout_stage_count is wrapped
