from areal.infra.scheduler.slurm import colocated_train_guard_fork_env


def test_fork_env_pins_gpu_and_strips_tms_preload():
    base_env = {
        "SOME_FLAG": "1",
        "LD_PRELOAD": "/x/torch_memory_saver_hook_mode_preload.abi3.so",
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }

    env = colocated_train_guard_fork_env(base_env, gpu_slot=5)

    assert env["CUDA_VISIBLE_DEVICES"] == "5"
    assert env["SOME_FLAG"] == "1"
    assert env["LD_PRELOAD"] == ""
    assert env["TMS_INIT_ENABLE"] == "0"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"
    assert base_env["LD_PRELOAD"].endswith(".so")


def test_fork_env_overrides_inherited_tms_even_when_absent_from_base():
    env = colocated_train_guard_fork_env({}, gpu_slot=0)

    assert env["CUDA_VISIBLE_DEVICES"] == "0"
    assert env["LD_PRELOAD"] == ""
    assert env["TMS_INIT_ENABLE"] == "0"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "0"
