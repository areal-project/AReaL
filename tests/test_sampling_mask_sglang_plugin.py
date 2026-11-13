# SPDX-License-Identifier: Apache-2.0
"""Registration tests plus opt-in execution of upstream sampler/plugin contracts.

Set AREAL_TEST_SGLANG_SOURCES to a directory containing v<version>/srt source
snapshots to run the three-version contract matrix without installing GPU stacks.
These tests execute upstream factories/hooks with CPU stand-ins for the runtime;
they are not a substitute for full SGLang GPU integration tests.
"""

import ast
import logging
import os
import sys
from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from areal.engine import sglang_sampling_mask_plugin as plugin


@pytest.fixture(autouse=True)
def isolated_registration(monkeypatch):
    monkeypatch.setattr(plugin, "_registered", False)
    monkeypatch.setenv("AREAL_SAMPLING_MASK_TRANSFER", "1")


def test_disabled_plugin_does_not_import_sglang(monkeypatch):
    monkeypatch.delenv("AREAL_SAMPLING_MASK_TRANSFER")
    monkeypatch.setattr(plugin, "integration_version", lambda: pytest.fail("imported"))
    plugin.register()


@pytest.mark.parametrize("installed", ["0.5.10.post1", "0.5.19", "0.5.20"])
def test_explicit_version_adapter_selected(monkeypatch, installed):
    monkeypatch.setattr(plugin, "version", lambda name: installed)
    assert plugin.integration_version() == installed


def test_unknown_version_fails_before_launch(monkeypatch):
    monkeypatch.setattr(plugin, "version", lambda name: "0.5.21")
    with pytest.raises(RuntimeError, match="no validated sampler adapter"):
        plugin.prepare_launch("sglang.launch_server")


@pytest.mark.parametrize("installed", ["0.5.19", "0.5.20"])
def test_native_launch_uses_installed_entrypoint_and_preserves_allowlist(
    monkeypatch, installed
):
    monkeypatch.setattr(plugin, "version", lambda name: installed)
    monkeypatch.setenv("SGLANG_PLUGINS", "other,areal_sampling_mask")
    ep = EntryPoint(plugin.PLUGIN, f"{plugin.__name__}:register", "sglang.srt.plugins")
    monkeypatch.setattr(plugin, "entry_points", lambda **kwargs: [ep])
    assert plugin.prepare_launch("sglang.launch_server") == "sglang.launch_server"
    assert os.environ["SGLANG_PLUGINS"] == "other,areal_sampling_mask"
    assert os.environ["NCCL_CUMEM_ENABLE"] == "1"


def test_native_launch_requires_package_metadata(monkeypatch):
    monkeypatch.setattr(plugin, "version", lambda name: "0.5.20")
    monkeypatch.delenv("SGLANG_PLUGINS", raising=False)
    monkeypatch.setattr(plugin, "entry_points", lambda **kwargs: [])
    with pytest.raises(RuntimeError, match="reinstall"):
        plugin.prepare_launch("sglang.launch_server")


def test_native_allowlist_exclusion_is_not_silent(monkeypatch):
    monkeypatch.setattr(plugin, "version", lambda name: "0.5.19")
    monkeypatch.setenv("SGLANG_PLUGINS", "another_plugin")
    with pytest.raises(RuntimeError, match="excludes required plugin"):
        plugin.prepare_launch("sglang.launch_server")


@pytest.mark.parametrize(
    "entry", ["sglang.launch_server", "areal.engine.awex.plugins.sglang_plugin"]
)
def test_legacy_launch_selects_spawn_wrapper(monkeypatch, entry):
    monkeypatch.setattr(plugin, "version", lambda name: "0.5.10.post1")
    assert plugin.prepare_launch(entry) == "areal.engine.sglang_sampling_mask_launcher"
    assert os.environ["AREAL_SAMPLING_MASK_ENTRYPOINT"] == entry


def test_registration_errors_cannot_be_swallowed_by_native_loader(monkeypatch):
    monkeypatch.setattr(plugin, "version", lambda name: "unsupported")
    with pytest.raises(SystemExit, match="Cannot initialize.*unsupported"):
        plugin.register()


def install_module(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(params=["0.5.10.post1", "0.5.19", "0.5.20"])
def upstream_runtime(request, monkeypatch):
    """Execute real upstream registry, constructor and hooks, isolating GPU deps."""
    installed = request.param
    source_root = os.environ.get("AREAL_TEST_SGLANG_SOURCES")
    if not source_root:
        pytest.skip("Set AREAL_TEST_SGLANG_SOURCES for upstream source contract tests")
    root = Path(source_root) / f"v{installed}" / "srt"
    assert (root / "layers/sampler.py").is_file()
    monkeypatch.setattr(plugin, "version", lambda name: installed)
    for package in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.managers",
        "sglang.srt.sampling",
        "sglang.srt.arg_groups",
    ):
        install_module(monkeypatch, package, __path__=[])
    settings = SimpleNamespace(
        sampling_backend=plugin.BACKEND,
        pp_size=1,
        attn_cp_size=1,
        speculative_algorithm=None,
        enable_nan_detection=False,
        rl_on_policy_target=None,
        enable_deterministic_inference=False,
    )
    choices = {"pytorch", "flashinfer", "ascend"}
    install_module(
        monkeypatch, "sglang.srt.arg_groups.choices", SAMPLING_BACKEND_CHOICES=choices
    )
    runtime = dict(
        get_exec=lambda: SimpleNamespace(
            kernel=settings,
            deterministic=settings,
            features=SimpleNamespace(sampling_mask_max_tokens=128),
        ),
        get_parallel=lambda: settings,
        get_spec=lambda: settings,
        get_server_args=lambda: settings,
    )
    install_module(monkeypatch, "sglang.srt.runtime_context", **runtime)

    class ServerArgs:
        def _handle_deterministic_inference(self):
            self.sampling_backend = "pytorch"
            self.other_deterministic_work = True

    def legacy_access():
        if installed != "0.5.10.post1":
            raise AssertionError("Do not use deprecated get_global_server_args")
        return settings

    install_module(
        monkeypatch,
        "sglang.srt.server_args",
        ServerArgs=ServerArgs,
        get_global_server_args=legacy_access,
        SAMPLING_BACKEND_CHOICES=choices,
    )

    class TokenizerManager:
        def _handle_batch_output(self, recv_obj):
            return recv_obj.customized_info

    if installed != "0.5.10.post1":
        tokenizer_source = root / "managers/tokenizer_manager.py"
        tree = ast.parse(tokenizer_source.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "TokenizerManager"
        )
        cls.bases = []
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            in ("_accumulate_request_meta_info", "update_request_meta_info")
        ]
        context = {}
        code = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(code), str(tokenizer_source), "exec"),
            context,
        )
        TokenizerManager = context["TokenizerManager"]
    install_module(
        monkeypatch,
        "sglang.srt.managers.tokenizer_manager",
        TokenizerManager=TokenizerManager,
    )
    install_module(
        monkeypatch, "sglang.srt.constants", HEALTH_CHECK_RID_PREFIX="health_"
    )

    @dataclass
    class SamplingBatchInfo:
        versions: list

        @classmethod
        def from_schedule_batch(cls, batch, vocab_size):
            return cls([])

        def filter_batch(self, indices, indices_device):
            pass

        def merge_batch(self, other):
            pass

        def copy_for_forward(self):
            return replace(self)

    install_module(
        monkeypatch,
        "sglang.srt.sampling.sampling_batch_info",
        SamplingBatchInfo=SamplingBatchInfo,
    )
    sampler = install_module(monkeypatch, "sglang.srt.layers.sampler")
    sampler.__dict__.update(
        torch=torch,
        nn=torch.nn,
        dist=torch.distributed,
        get_global_server_args=legacy_access,
        **runtime,
        get_tp_group=lambda: SimpleNamespace(device_group="test_tp"),
        is_dp_attention_enabled=lambda: False,
        OutputLogprobProcessor=lambda: None,
        SYNC_TOKEN_IDS_ACROSS_TP=False,
        logger=logging.getLogger("UpstreamContract"),
        _CUSTOM_SAMPLER_FACTORIES={},
        _BUILT_IN_SAMPLING_BACKENDS={"pytorch", "flashinfer", "ascend"},
    )
    tree = ast.parse((root / "layers/sampler.py").read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Sampler":
            node.body = [
                method
                for method in node.body
                if isinstance(method, ast.FunctionDef)
                and method.name in ("__init__", "_sync_token_ids_across_tp")
            ]
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in (
            "register_sampler_backend",
            "create_sampler",
        ):
            selected.append(node)
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(
        compile(
            ast.fix_missing_locations(code), str(root / "layers/sampler.py"), "exec"
        ),
        sampler.__dict__,
    )

    def handle_deterministic_inference(args):
        args.sampling_backend = "pytorch"
        args.other_deterministic_work = True

    attention = install_module(
        monkeypatch,
        "sglang.srt.arg_groups.attention_hook",
        handle_deterministic_inference=handle_deterministic_inference,
    )

    def declare(args, source, **values):
        for key, value in values.items():
            setattr(args, key, value)

    install_module(
        monkeypatch,
        "sglang.srt.arg_groups.overrides",
        resolving_view=lambda args: args,
        declare_resolution=declare,
    )
    if installed != "0.5.10.post1":
        install_module(
            monkeypatch,
            "sglang.srt.environ",
            envs=SimpleNamespace(
                SGLANG_PLUGINS=SimpleNamespace(get=lambda: None),
                SGLANG_PLATFORM=SimpleNamespace(get=lambda: None),
            ),
        )
        loader = install_module(monkeypatch, "sglang.srt.plugins", __path__=[])
        registry = install_module(monkeypatch, "sglang.srt.plugins.hook_registry")
        exec(
            compile(
                (root / "plugins/hook_registry.py").read_text(),
                str(root / "plugins/hook_registry.py"),
                "exec",
            ),
            registry.__dict__,
        )
        exec(
            compile(
                (root / "plugins/__init__.py").read_text(),
                str(root / "plugins/__init__.py"),
                "exec",
            ),
            loader.__dict__,
        )
        ep = SimpleNamespace(
            name=plugin.PLUGIN,
            dist=None,
            value=f"{plugin.__name__}:register",
            load=lambda: plugin.register,
        )
        entries = [ep]
        if installed == "0.5.19":

            def register_later_plugin():
                def later_hook(previous, args):
                    args.other_plugin_ran = True
                    return previous(args)

                registry.HookRegistry.register(
                    "sglang.srt.arg_groups.attention_hook.handle_deterministic_inference",
                    later_hook,
                    registry.HookType.AROUND,
                )

            entries.append(
                SimpleNamespace(
                    name="later_plugin",
                    dist=None,
                    value="test:register",
                    load=lambda: register_later_plugin,
                )
            )
        monkeypatch.setattr(loader, "entry_points", lambda **kwargs: entries)
        if installed == "0.5.20":
            hooks = install_module(
                monkeypatch, "sglang.srt.arg_groups.resolution_hooks"
            )
            exec(
                compile(
                    (root / "arg_groups/resolution_hooks.py").read_text(),
                    str(root / "arg_groups/resolution_hooks.py"),
                    "exec",
                ),
                hooks.__dict__,
            )
        loader.load_plugins()
    else:
        plugin.register()
    return SimpleNamespace(
        version=installed,
        settings=settings,
        sampler=sampler,
        attention=attention,
        args_cls=ServerArgs,
        tokenizer_cls=TokenizerManager,
        batch_cls=SamplingBatchInfo,
    )


def test_upstream_factory_instantiates_and_samples_restricted_policy(
    upstream_runtime, monkeypatch
):
    runtime = upstream_runtime
    from areal.infra.sampling_mask import service

    rows = []

    class Producer:
        def __init__(self, *args, **kwargs):
            pass

        def publish(self, ids, counts, tokens, versions):
            rows.append((ids, counts, tokens, versions))
            return [[1, 2, 3, 4, 5]]

    monkeypatch.setattr(service, "MaskProducer", Producer)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setenv("AREAL_SAMPLING_MASK_TOP_K", "3")
    monkeypatch.setenv("AREAL_SAMPLING_MASK_RECEIVERS", "1")
    sampler = runtime.sampler.create_sampler(plugin.BACKEND)
    output = SimpleNamespace(next_token_logits=torch.tensor([[3.0, 2.0, 1.0, 0.0]]))
    info = SimpleNamespace(
        has_custom_logit_processor=False,
        grammars=None,
        temperatures=torch.tensor([1.0]),
        top_ps=torch.tensor([0.95]),
        top_ks=torch.tensor([3]),
        sampling_seed=None,
        _areal_mask_versions=[7],
    )
    token = sampler(output, info, True, [], [], torch.tensor([0]))
    expected = torch.log_softmax(torch.tensor([3.0, 2.0, 1.0]), dim=0)[token.long()]
    torch.testing.assert_close(output.next_token_logprobs, expected, atol=1e-6, rtol=0)
    assert rows[0][1].tolist() == [3]
    assert rows[0][3] == [7]
    assert output.customized_info["sampling_mask_refs"] == [[1, 2, 3, 4, 5]]
    assert plugin.BACKEND in runtime.sampler._CUSTOM_SAMPLER_FACTORIES


@pytest.mark.parametrize("backend", [plugin.BACKEND, "flashinfer", "pytorch"])
def test_upstream_deterministic_resolution_preserves_only_our_backend(
    upstream_runtime, backend
):
    runtime = upstream_runtime
    args = runtime.args_cls()
    args.sampling_backend = backend
    if runtime.version == "0.5.10.post1":
        args._handle_deterministic_inference()
    elif runtime.version == "0.5.19":
        runtime.attention.handle_deterministic_inference(args)
    else:
        from sglang.srt.arg_groups.resolution_hooks import run_hook

        run_hook(runtime.attention.handle_deterministic_inference, args)
    if runtime.version == "0.5.19":
        assert args.other_plugin_ran
    assert args.other_deterministic_work
    assert args.sampling_backend == (
        plugin.BACKEND if backend == plugin.BACKEND else "pytorch"
    )


def test_metadata_accumulates_through_eos(upstream_runtime):
    runtime = upstream_runtime
    tokenizer = runtime.tokenizer_cls()
    tokenizer.rid_to_state = {"a": SimpleNamespace(customized_info_accumulated={})}
    for index in range(3):
        output = SimpleNamespace(
            rids=["a"],
            customized_info={"sampling_mask_refs": [[[index]]], "other": [index]},
        )
        if runtime.version == "0.5.10.post1":
            result = tokenizer._handle_batch_output(output)
            assert result["sampling_mask_refs"] == [[[i] for i in range(index + 1)]]
            assert result["other"] == [index]
        else:
            result = {}
            tokenizer.update_request_meta_info(
                result,
                tokenizer.rid_to_state["a"],
                {"sampling_mask_refs": [[[index]]]},
                0,
                {"type": "stop"} if index == 2 else None,
            )
            assert result["sampling_mask_refs"] == [[i] for i in range(index + 1)]


def test_upstream_dataclass_copy_retains_mask_versions(upstream_runtime):
    original = upstream_runtime.batch_cls([])
    original._areal_mask_versions = [2, 3]
    copied = original.copy_for_forward()
    assert copied._areal_mask_versions == [2, 3]
    assert copied._areal_mask_versions is not original._areal_mask_versions


@pytest.mark.parametrize(
    "unsupported", ["grpc_mode", "encoder_only", "use_ray", "tokenizer_worker_num"]
)
def test_legacy_launcher_rejects_workers_without_initializers(monkeypatch, unsupported):
    from areal.engine.sglang_sampling_mask_launcher import main

    monkeypatch.setenv("AREAL_SAMPLING_MASK_ENTRYPOINT", "sglang.launch_server")
    monkeypatch.setattr(plugin, "register", lambda: None)
    args = SimpleNamespace(
        grpc_mode=False, encoder_only=False, use_ray=False, tokenizer_worker_num=1
    )
    setattr(args, unsupported, 2 if unsupported == "tokenizer_worker_num" else True)
    install_module(
        monkeypatch,
        "sglang.srt.entrypoints.http_server",
        launch_server=lambda *a, **kw: pytest.fail("launched unsupported server"),
    )
    install_module(
        monkeypatch, "sglang.srt.server_args", prepare_server_args=lambda argv: args
    )
    install_module(
        monkeypatch, "sglang.srt.utils", kill_process_tree=lambda *a, **kw: None
    )
    with pytest.raises(ValueError, match="Legacy sampling mask launcher"):
        main()


@pytest.mark.parametrize(
    "entry", ["sglang.launch_server", "areal.engine.awex.plugins.sglang_plugin"]
)
@pytest.mark.parametrize("hip_version", [None, "7.15.0"])
def test_legacy_launcher_starts_http_with_real_backend_normalizer(
    monkeypatch, entry, hip_version
):
    """Exercise successful launch, including the AWEX helper dependency."""
    from areal.engine.sglang_sampling_mask_launcher import main, run_scheduler_process

    monkeypatch.setenv("AREAL_SAMPLING_MASK_ENTRYPOINT", entry)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    events = []
    monkeypatch.setattr(plugin, "register", lambda: events.append("register"))
    install_module(
        monkeypatch,
        "areal.engine.awex.plugins.sglang_plugin",
        _load_sglang_plugins_if_available=lambda: events.append("awex"),
    )
    args = SimpleNamespace(
        grpc_mode=False,
        encoder_only=False,
        use_ray=False,
        tokenizer_worker_num=1,
        attention_backend="fa3",
        enable_deterministic_inference=False,
        skip_server_warmup=False,
    )
    launched = []
    install_module(
        monkeypatch,
        "sglang.srt.entrypoints.http_server",
        launch_server=lambda *a, **kw: launched.append((a, kw)),
    )
    install_module(
        monkeypatch, "sglang.srt.server_args", prepare_server_args=lambda argv: args
    )
    install_module(
        monkeypatch,
        "sglang.srt.utils",
        kill_process_tree=lambda *a, **kw: events.append("cleanup"),
    )

    main()

    awex = entry == "areal.engine.awex.plugins.sglang_plugin"
    assert events == (["awex"] if awex else []) + ["register", "cleanup"]
    assert args.attention_backend == ("aiter" if awex and hip_version else "fa3")
    assert args.skip_server_warmup is awex
    assert launched == [
        ((args,), {"run_scheduler_process_func": run_scheduler_process})
    ]
