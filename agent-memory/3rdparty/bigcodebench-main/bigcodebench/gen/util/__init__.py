import os
import time
import sys
import types
import unittest
import tempfile
import multiprocessing
from multiprocessing import Array, Value, Manager
from bigcodebench.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
    safe_environment,
    TIMEOUT_LIMIT,
)


def trusted_exec(code, test_code, task_id, max_as_limit, max_data_limit, max_stack_limit, times):
    """Execute trusted code in place."""
    # Specify a unique cache dir by modifying XDG_CONFIG_HOME
    old_xdg = os.environ.get("XDG_CONFIG_HOME")
    temp_xdg = tempfile.mkdtemp(prefix="xdg_config_")
    os.environ["XDG_CONFIG_HOME"] = temp_xdg

    try:
        with create_tempdir():
            import shutil
            import builtins

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir
            module_name = "__test__"
            new_module = types.ModuleType(module_name)

            reliability_guard(max_as_limit, max_data_limit, max_stack_limit)

            # Set necessary attributes for the module
            new_module.__dict__.update({
                '__builtins__': builtins,
                '__file__': f"{module_name}.py",
                '__package__': None,
                '__doc__': None,
                'sys': sys,
                'os': os,
                'environ': os.environ,
            })

            # Combine the user code and the test code
            full_code = code + "\n" + test_code

            # Compile and execute the combined code within the new module
            exec(compile(full_code, f"{module_name}.py", 'exec'),
                 new_module.__dict__)
            sys.modules[module_name] = new_module
            TestCases = getattr(new_module, 'TestCases')
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(TestCases)
            test_result = unittest.TestResult()
            start = time.time()
            with safe_environment(), swallow_io(), time_limit(seconds=TIMEOUT_LIMIT):
                suite.run(test_result)

            errors = test_result.failures + test_result.errors
            if len(errors) > 0:
                print(errors)
                times.value = -1
            else:
                times.value = time.time() - start

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    finally:
        # Restore the original environment variable
        if old_xdg is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = old_xdg
        shutil.rmtree(temp_xdg, ignore_errors=True)


def trusted_check_exec(code, inputs):
    """Check trusted_exec success."""
    try:
        with time_limit(seconds=TIMEOUT_LIMIT):
            trusted_exec(code, inputs)
    except Exception:
        return False
    return True


def trusted_check(
    code: str,
    test_code: str,
    task_id: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    min_time_limit: float = 10,
):
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), min_time_limit) + 1
    # shared memory objects
    times = Value("d", -1)
    manager = Manager()

    p = multiprocessing.Process(
        target=trusted_exec,
        args=(
            code,
            test_code,
            task_id,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            times,
        ),
    )
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    if times.value == -1:
        times = None
    else:
        times = times.value
    
    return {"task_id": task_id, "time": times}