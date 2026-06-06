"""C++ code runner with time and memory limits."""

import math
import os
import re
import subprocess
import sys
import tempfile
import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Try to import resource module (not available on Windows)
try:
    import resource
    _has_resource = True
except ImportError:
    _has_resource = False
    resource = None

# Try to import signal module for signal constants
try:
    import signal
    _has_signal = True
except ImportError:
    _has_signal = False
    signal = None


def preprocess_cpp_code(code: str) -> str:
    """Preprocess C++ code to fix common issues.
    
    This function replaces problematic includes like bits/stdc++.h with
    specific standard library headers to avoid compilation errors.
    
    Args:
        code: Original C++ source code
        
    Returns:
        Preprocessed C++ source code
    """
    # Replace bits/stdc++.h with commonly used headers
    # This avoids conflicts with valarray operators that can cause compilation errors
    common_headers = """#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <queue>
#include <stack>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <fstream>"""
    
    # Pattern to match various forms of bits/stdc++.h include
    # Match both angle brackets and quotes, with optional whitespace
    pattern = r'#include\s*[<"]bits/stdc\+\+\.h[>"]'
    
    processed_code = code
    if re.search(pattern, processed_code, re.IGNORECASE):
        # Replace with common headers
        processed_code = re.sub(
            pattern,
            common_headers,
            processed_code,
            flags=re.IGNORECASE
        )
    
    return processed_code


def compile_cpp(source_file: Path, executable_file: Path, timeout: int = 30) -> Tuple[bool, Dict[str, Any]]:
    """Compile C++ source code.
    
    Args:
        source_file: Path to C++ source file
        executable_file: Path to output executable file
        timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (success: bool, metadata: dict)
    """
    metadata = {}
    compile_cmd = ["g++", "-O2", "-std=c++17", "-o", str(executable_file), str(source_file)]
    
    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if compile_result.returncode != 0:
            metadata["error"] = "Compilation Error"
            metadata["error_message"] = compile_result.stderr
            metadata["stderr"] = compile_result.stderr
            metadata["stdout"] = compile_result.stdout
            return False, metadata
        
        return True, metadata
        
    except subprocess.TimeoutExpired:
        metadata["error"] = "Compilation Timeout"
        metadata["error_message"] = f"Compilation exceeded {timeout} seconds"
        return False, metadata
    except FileNotFoundError:
        metadata["error"] = "Compiler Not Found"
        metadata["error_message"] = "g++ compiler not found. Please ensure g++ is installed."
        return False, metadata
    except Exception as e:
        metadata["error"] = "Compilation Exception"
        metadata["error_message"] = str(e)
        return False, metadata


def run_with_limits(
    executable_file: Path,
    input_data: str,
    time_limit_sec: float,
    memory_limit_mb: int,
) -> Tuple[subprocess.CompletedProcess, Dict[str, Any]]:
    """Run executable with time and memory limits.
    
    Args:
        executable_file: Path to executable file
        input_data: Input data as string (stdin)
        time_limit_sec: CPU time limit in seconds
        memory_limit_mb: Memory limit in MB
        
    Returns:
        Tuple of (CompletedProcess, metadata dict)
    """
    metadata = {}
    memory_limit_bytes = memory_limit_mb * 1024 * 1024 if memory_limit_mb > 0 else 0
    
    def set_limits():
        """Set resource limits for the subprocess."""
        if not _has_resource or resource is None:
            return
        try:
            # Set CPU time limit (soft and hard limit)
            # RLIMIT_CPU: maximum CPU time in seconds
            if time_limit_sec is not None and time_limit_sec != float('inf') and time_limit_sec > 0:
                time_limit_int = int(time_limit_sec)
                resource.setrlimit(resource.RLIMIT_CPU, (time_limit_int, time_limit_int + 1))
            
            # Set memory limit (virtual memory/address space)
            # RLIMIT_AS: maximum virtual memory in bytes
            if memory_limit_bytes > 0:
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        except ValueError as e:
            # When trying to set a value beyond system limits
            metadata["resource_limit_warning"] = f"Resource limit setting failed: {e}"
        except (OSError, AttributeError) as e:
            # System doesn't support it or resource module not available
            metadata["resource_limit_warning"] = f"Resource limit setting error: {e}"
    
    # Measure actual execution time
    start_time = time.perf_counter()
    
    try:
        # Run with timeout as a safety net (longer than CPU limit to account for multi-threading overhead)
        # Increased buffer from 2s to 10s to handle multi-threading timing fluctuations
        if time_limit_sec is None or time_limit_sec == float('inf'):
            timeout_safety = None
        elif time_limit_sec > 0:
            # Add 10 seconds buffer (or 20% of time limit, whichever is larger) to handle multi-threading overhead
            buffer_sec = max(10.0, time_limit_sec * 0.2)
            timeout_safety = time_limit_sec + buffer_sec
        else:
            timeout_safety = None
        
        run_result = subprocess.run(
            [str(executable_file)],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout_safety,
            preexec_fn=set_limits if (_has_resource and resource is not None) else None,
        )
        
        end_time = time.perf_counter()
        actual_runtime_sec = end_time - start_time
        metadata["actual_runtime_sec"] = actual_runtime_sec
        metadata["actual_runtime_ms"] = actual_runtime_sec * 1000
        
        # Try to get CPU time from resource usage (if available)
        if _has_resource and resource is not None:
            try:
                # Get resource usage for the current process
                # Note: This gets usage for the parent process, not the subprocess
                # For subprocess CPU time, we'd need to use resource.getrusage(resource.RUSAGE_CHILDREN)
                # but that requires tracking child processes
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                cpu_time_sec = usage.ru_utime + usage.ru_stime  # user time + system time
                metadata["cpu_time_sec"] = cpu_time_sec
                metadata["cpu_time_ms"] = cpu_time_sec * 1000
            except (OSError, AttributeError):
                # Resource usage not available or not supported
                pass
        
        # Check if actual runtime exceeds time limit
        # This catches cases where wall-clock time exceeds limit even if CPU time doesn't
        # Add tolerance for multi-threading overhead (10% or 0.5s, whichever is larger)
        if time_limit_sec is not None and time_limit_sec != float('inf') and time_limit_sec > 0:
            tolerance_sec = max(0.2, time_limit_sec * 0.1)  # 10% tolerance or 0.2s minimum
            effective_limit = time_limit_sec + tolerance_sec
            if actual_runtime_sec > effective_limit:
                metadata["time_limit_exceeded"] = True
                metadata["time_limit_exceeded_by_actual_runtime"] = True
                metadata["time_limit_sec"] = time_limit_sec
                metadata["tolerance_sec"] = tolerance_sec
                metadata["effective_limit_sec"] = effective_limit
                metadata["exceeded_by_sec"] = actual_runtime_sec - effective_limit
        
        # Analyze the result
        if run_result.returncode < 0:
            signal_number = -run_result.returncode
            
            # SIGXCPU (24): CPU time limit exceeded
            if _has_signal and signal_number == signal.SIGXCPU:
                metadata["time_limit_exceeded"] = True
                metadata["signal"] = "SIGXCPU"
            elif signal_number == 24:  # Fallback: SIGXCPU is typically 24 on Linux
                metadata["time_limit_exceeded"] = True
                metadata["signal"] = "SIGXCPU"
            
            # SIGKILL (9) or SIGSEGV (11): Often indicates memory limit exceeded
            elif signal_number == 9 or signal_number == 11:
                metadata["memory_limit_exceeded"] = True
                metadata["signal"] = "SIGKILL" if signal_number == 9 else "SIGSEGV"
            
            else:
                metadata["signal"] = signal_number
                metadata["runtime_error"] = True
        
        elif run_result.returncode != 0:
            metadata["runtime_error"] = True
            metadata["returncode"] = run_result.returncode
        
        return run_result, metadata
        
    except subprocess.TimeoutExpired:
        end_time = time.perf_counter()
        actual_runtime_sec = end_time - start_time
        metadata["timeout"] = True
        metadata["timeout_limit"] = timeout_safety
        metadata["actual_runtime_sec"] = actual_runtime_sec
        metadata["actual_runtime_ms"] = actual_runtime_sec * 1000
        
        # Check if actual runtime exceeds time limit
        # Add tolerance for multi-threading overhead (10% or 0.5s, whichever is larger)
        if time_limit_sec is not None and time_limit_sec != float('inf') and time_limit_sec > 0:
            tolerance_sec = max(0.5, time_limit_sec * 0.1)  # 10% tolerance or 0.5s minimum
            effective_limit = time_limit_sec + tolerance_sec
            if actual_runtime_sec > effective_limit:
                metadata["time_limit_exceeded"] = True
                metadata["time_limit_exceeded_by_actual_runtime"] = True
                metadata["time_limit_sec"] = time_limit_sec
                metadata["tolerance_sec"] = tolerance_sec
                metadata["effective_limit_sec"] = effective_limit
                metadata["exceeded_by_sec"] = actual_runtime_sec - effective_limit
        
        # Create a mock CompletedProcess for timeout
        return subprocess.CompletedProcess(
            args=[str(executable_file)],
            returncode=-1,
            stdout="",
            stderr="",
        ), metadata
    except Exception as e:
        end_time = time.perf_counter()
        actual_runtime_sec = end_time - start_time
        metadata["exception"] = str(e)
        metadata["exception_type"] = type(e).__name__
        metadata["actual_runtime_sec"] = actual_runtime_sec
        metadata["actual_runtime_ms"] = actual_runtime_sec * 1000
        # Create a mock CompletedProcess for exception
        return subprocess.CompletedProcess(
            args=[str(executable_file)],
            returncode=-1,
            stdout="",
            stderr=str(e),
        ), metadata


def compare_output(actual_output: str, expected_output: str) -> bool:
    """Compare actual output with expected output.
    
    Supports:
    - Exact match
    - Line-by-line comparison (ignoring trailing whitespace)
    - Numeric comparison for floating point values
    
    Args:
        actual_output: Actual output from program
        expected_output: Expected output
        
    Returns:
        True if outputs match, False otherwise
    """
    actual = actual_output.strip()
    expected = str(expected_output).strip()
    
    # print("actual:", actual)
    # print("expected:", expected)
    # Try exact match first
    if actual == expected:
        return True
    
    # Try line-by-line comparison (ignoring trailing whitespace)
    actual_lines = [line.strip() for line in actual.splitlines() if line.strip()]
    expected_lines = [line.strip() for line in expected.splitlines() if line.strip()]
    
    if actual_lines == expected_lines:
        return True
    
    # Try numeric comparison for floating point
    try:
        actual_nums = [float(x) for x in actual_lines]
        expected_nums = [float(x) for x in expected_lines]
        if len(actual_nums) == len(expected_nums):
            if all(
                math.isclose(a, e, rel_tol=1e-9, abs_tol=1e-9)
                for a, e in zip(actual_nums, expected_nums)
            ):
                return True
    except (ValueError, TypeError):
        pass
    
    return False


def run_cpp_tests(
    solution: str,
    inputs: List[str],
    outputs: List[str],
    timeout_ms: int = 2000,
    memory_limit_mb: int = 1024,
    compile_timeout: int = 30,
) -> Tuple[List[bool], Dict[str, Any]]:
    """Compile and run C++ code with test cases, applying time and memory limits.
    
    Args:
        solution: C++ source code
        inputs: List of input strings for test cases
        outputs: List of expected output strings for test cases
        timeout_ms: Time limit in milliseconds (0 means no limit)
        memory_limit_mb: Memory limit in MB (0 means no limit)
        compile_timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (results list, metadata dict)
    """
    results = []
    metadata = {}
    
    # Convert timeout from milliseconds to seconds
    time_limit_sec = timeout_ms / 1000.0 if timeout_ms > 0 else None
    
    # Create temporary directory for compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_file = Path(tmpdir) / "solution.cpp"
        exe_file = Path(tmpdir) / "solution"
        
        # Preprocess C++ code to fix common issues (e.g., replace bits/stdc++.h)
        processed_solution = preprocess_cpp_code(solution)
        
        # Write C++ code to file
        cpp_file.write_text(processed_solution, encoding="utf-8")
        
        # Compile C++ code
        compile_success, compile_metadata = compile_cpp(cpp_file, exe_file, compile_timeout)
        metadata.update(compile_metadata)
        
        if not compile_success:
            return [False] * len(inputs), metadata
        
        # Run tests
        for idx, (input_data, expected_output) in enumerate(zip(inputs, outputs)):
            test_metadata = {}
            
            try:
                # Execute with time and memory limits
                run_result, run_metadata = run_with_limits(
                    exe_file,
                    input_data,
                    time_limit_sec,
                    memory_limit_mb,
                )
                
                test_metadata.update(run_metadata)
                
                # Check for errors
                if run_result.returncode != 0:
                    results.append(False)
                    test_metadata["error"] = True

                    if "time_limit_exceeded" in run_metadata or "timeout" in run_metadata:
                        test_metadata["error_type"] = "Time Limit Exceeded"
                        if "timeout" in run_metadata:
                            test_metadata["timeout_limit_ms"] = timeout_ms
                    elif "memory_limit_exceeded" in run_metadata:
                        test_metadata["error_type"] = "Memory Limit Exceeded"
                        test_metadata["memory_limit_mb"] = memory_limit_mb
                    else:
                        test_metadata["error_type"] = "Runtime Error"
                    
                    test_metadata["stderr"] = run_result.stderr
                    test_metadata["returncode"] = run_result.returncode
                    
                else:
                    # Compare output
                    actual_output = run_result.stdout
                    is_correct = compare_output(actual_output, expected_output)
                    results.append(is_correct)
                    
                    if not is_correct:
                        test_metadata["output_mismatch"] = True
                        test_metadata["actual_output"] = actual_output
                        test_metadata["expected_output"] = expected_output
                
                # Store test metadata
                for key, value in test_metadata.items():
                    metadata[f"test_{idx}_{key}"] = value
                
            except Exception as e:
                results.append(False)
                metadata[f"test_{idx}_exception"] = str(e)
                metadata[f"test_{idx}_exception_type"] = type(e).__name__
                if "error_message" not in metadata:
                    metadata["error_message"] = f"Exception in test {idx}: {str(e)}"
    
    return results, metadata


def parse_time_limit(time_str: str) -> float:
    """Parse time limit string (e.g., '2s', '1000ms') to seconds.
    
    Args:
        time_str: Time limit string
        
    Returns:
        Time limit in seconds
    """
    time_str = time_str.strip().lower()
    if time_str.endswith('ms'):
        return float(time_str[:-2]) / 1000.0
    elif time_str.endswith('s'):
        return float(time_str[:-1])
    elif time_str.endswith('m'):
        return float(time_str[:-1]) * 60.0
    else:
        # Assume seconds if no unit
        return float(time_str)


def parse_memory_limit(mem_str: str) -> int:
    """Parse memory limit string (e.g., '512m', '1g') to MB.
    
    Args:
        mem_str: Memory limit string
        
    Returns:
        Memory limit in MB
    """
    mem_str = mem_str.strip().lower()
    if mem_str.endswith('g'):
        return int(float(mem_str[:-1]) * 1024)
    elif mem_str.endswith('m'):
        return int(float(mem_str[:-1]))
    elif mem_str.endswith('k'):
        return int(float(mem_str[:-1]) / 1024)
    else:
        # Assume MB if no unit
        return int(float(mem_str))


def _auto_extract_zip(data_dir: str, problem_id: str) -> bool:
    """Auto-extract a zip file for a problem if directory doesn't exist.

    Args:
        data_dir: Root data directory
        problem_id: Problem ID (subdirectory name)

    Returns:
        True if extraction succeeded or directory already exists, False otherwise
    """
    import zipfile
    problem_dir = Path(data_dir) / problem_id
    if problem_dir.exists() and (problem_dir / "config.yaml").exists():
        return True

    zip_file = Path(data_dir) / f"{problem_id}.zip"
    if not zip_file.exists():
        return False

    try:
        problem_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(problem_dir)
        return True
    except Exception:
        return False


def load_problem_config(data_dir: str, problem_id: str) -> Dict[str, Any]:
    """Load problem configuration from data directory.

    If the problem directory doesn't exist but a zip file does, auto-extracts it.

    Args:
        data_dir: Root data directory
        problem_id: Problem ID (subdirectory name)

    Returns:
        Configuration dictionary
    """
    problem_dir = Path(data_dir) / problem_id
    config_file = problem_dir / "config.yaml"

    if not config_file.exists():
        # Try auto-extracting from zip
        if _auto_extract_zip(data_dir, problem_id):
            config_file = problem_dir / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Parse time and memory limits
    if 'time_limit' in config:
        config['time_limit_sec'] = parse_time_limit(config['time_limit'])
    if 'memory_limit' in config:
        config['memory_limit_mb'] = parse_memory_limit(config['memory_limit'])

    return config


def load_test_cases(data_dir: str, problem_id: str, config: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str]]:
    """Load test cases from data directory.
    
    Args:
        data_dir: Root data directory
        problem_id: Problem ID (subdirectory name)
        config: Optional configuration dict (if not provided, will be loaded)
        
    Returns:
        Tuple of (inputs list, outputs list)
    """
    if config is None:
        config = load_problem_config(data_dir, problem_id)
    
    problem_dir = Path(data_dir) / problem_id
    testdata_dir = problem_dir / "testdata"
    
    if not testdata_dir.exists():
        raise FileNotFoundError(f"Testdata directory not found: {testdata_dir}")
    
    # Get file suffixes from config
    input_suffix = config.get('input_suffix', '.in')
    output_suffix = config.get('output_suffix', '.ans')
    input_prefix = config.get('input_prefix', '')
    output_prefix = config.get('output_prefix', '')
    
    # Find all test case files
    input_files = sorted(testdata_dir.glob(f"{input_prefix}*{input_suffix}"))
    
    inputs = []
    outputs = []
    
    for input_file in input_files:
        # Get corresponding output file
        # Extract base name (filename without prefix and suffix)
        base_name = input_file.name
        if input_prefix:
            base_name = base_name[len(input_prefix):] if base_name.startswith(input_prefix) else base_name
        if base_name.endswith(input_suffix):
            base_name = base_name[:-len(input_suffix)]
        
        output_file = testdata_dir / f"{output_prefix}{base_name}{output_suffix}"
        
        if not output_file.exists():
            continue
        
        # Read input and output
        with open(input_file, 'r', encoding='utf-8') as f:
            inputs.append(f.read())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            outputs.append(f.read())
    
    return inputs, outputs


def find_testlib() -> Optional[Path]:
    """Find testlib.h in common locations or from environment variable.
    
    Returns:
        Path to testlib.h if found, None otherwise
    """
    # Check environment variable first
    testlib_env = os.environ.get("TESTLIB_PATH")
    if testlib_env:
        testlib_path = Path(testlib_env)
        if testlib_path.is_file() and testlib_path.name == "testlib.h":
            return testlib_path
        elif testlib_path.is_dir():
            testlib_file = testlib_path / "testlib.h"
            if testlib_file.exists():
                return testlib_file
    
    # Try common locations
    testlib_paths = [
        # System locations
        Path("/usr/include/testlib.h"),
        Path("/usr/local/include/testlib.h"),
        # Project locations (relative to this file)
        Path(__file__).parent.parent.parent / "testlib" / "testlib.h",
        Path(__file__).parent.parent.parent.parent / "testlib" / "testlib.h",
        
    ]
    
    for path in testlib_paths:
        if path.exists():
            return path
    
    print("no testlib in: ", testlib_paths)
    return None


def compile_checker(checker_file: Path, checker_exe: Path, timeout: int = 30) -> Tuple[bool, Dict[str, Any]]:
    """Compile checker (testlib-based).
    
    Uses a shared testlib from a fixed location instead of requiring testlib
    in each problem directory.
    
    Args:
        checker_file: Path to checker.cpp
        checker_exe: Path to output executable
        timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (success: bool, metadata: dict)
    """
    metadata = {}
    
    # Find testlib.h
    testlib_h = find_testlib()
    
    if testlib_h is None:
        metadata["warning"] = "testlib.h not found, trying to compile without it"
        # Try to compile without explicit testlib path (might be in system include)
        compile_cmd = ["g++", "-O2", "-std=c++17", "-o", str(checker_exe), str(checker_file)]
    else:
        # Build compilation command with testlib
        testlib_dir = testlib_h.parent
        
        # Check if testlib.cpp exists (some testlib implementations need it)
        testlib_cpp = testlib_dir / "testlib.cpp"
        
        compile_cmd = [
            "g++", "-O2", "-std=c++17",
            "-I", str(testlib_dir),
            "-o", str(checker_exe),
        ]
        
        # Add testlib.cpp if it exists
        if testlib_cpp.exists():
            compile_cmd.append(str(testlib_cpp))
        
        # Add checker source file
        compile_cmd.append(str(checker_file))
        
        metadata["testlib_path"] = str(testlib_h)
        if testlib_cpp.exists():
            metadata["testlib_cpp"] = str(testlib_cpp)
    
    try:
        compile_result = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if compile_result.returncode != 0:
            metadata["error"] = "Checker Compilation Error"
            metadata["error_message"] = compile_result.stderr
            metadata["stderr"] = compile_result.stderr
            metadata["stdout"] = compile_result.stdout
            return False, metadata
        
        return True, metadata
        
    except subprocess.TimeoutExpired:
        metadata["error"] = "Checker Compilation Timeout"
        metadata["error_message"] = f"Checker compilation exceeded {timeout} seconds"
        return False, metadata
    except FileNotFoundError:
        metadata["error"] = "Compiler Not Found"
        metadata["error_message"] = "g++ compiler not found. Please ensure g++ is installed."
        return False, metadata
    except Exception as e:
        metadata["error"] = "Checker Compilation Exception"
        metadata["error_message"] = str(e)
        return False, metadata


def run_with_checker(
    solution_exe: Path,
    checker_exe: Path,
    input_data: str,
    expected_output: str,
    time_limit_sec: float,
    memory_limit_mb: int,
    max_retries: int = 4,
    checker_timeout: float = 60.0,  # Increased from 30.0 to 60.0 to handle multi-threading overhead
) -> Tuple[bool, Dict[str, Any]]:
    """Run solution and verify output using checker with retry mechanism.
    
    Args:
        solution_exe: Path to solution executable
        checker_exe: Path to checker executable
        input_data: Input data as string
        expected_output: Expected output as string
        time_limit_sec: CPU time limit in seconds
        memory_limit_mb: Memory limit in MB
        max_retries: Maximum number of retries on timeout (default: 2)
        checker_timeout: Timeout for checker execution in seconds (default: 30.0)
        
    Returns:
        Tuple of (is_correct: bool, metadata: dict)
    """
    metadata = {}
    metadata["checker_timeout_sec"] = checker_timeout
    
    # Retry loop for handling transient timeout issues
    for attempt in range(max_retries + 1):
        if attempt > 0:
            metadata[f"retry_attempt_{attempt}"] = True
            # Small delay before retry to avoid immediate re-execution
            time.sleep(0.1)
        
        try:
            # Create temporary files for input, output, and answer
            with tempfile.TemporaryDirectory() as tmpdir:
                input_file = Path(tmpdir) / "input.txt"
                output_file = Path(tmpdir) / "output.txt"
                answer_file = Path(tmpdir) / "answer.txt"
                
                # Write input and expected output
                input_file.write_text(input_data, encoding='utf-8')
                answer_file.write_text(expected_output, encoding='utf-8')
                
                
                # Run solution first
                run_result, run_metadata = run_with_limits(
                    solution_exe,
                    input_data,
                    time_limit_sec,
                    memory_limit_mb,
                )
                # print("result: ", run_result)
                # print("metadata: ", run_metadata)
                
                metadata.update(run_metadata)
                
                # Check if solution had timeout (subprocess timeout)
                # This can happen due to multi-threading issues or actual timeout
                if run_metadata.get("time_limit_exceeded", False) is True:
                    timeout_limit = run_metadata.get("timeout_limit")
                    metadata[f"retries_{attempt}_result"] = "Time Out"
                    if attempt < max_retries:
                        continue
                    metadata["user_code_timeout"] = True# Retry if we haven't exceeded max retries (could be transient multi-threading issue)
                    metadata["timeout_source"] = "user_code"
                    metadata["error"] = "TimeLimitExceeded"
                    metadata["error_type"] = "TimeLimitExceeded"
                    if timeout_limit:
                        metadata["timeout_message"] = f"User code subprocess timeout ({timeout_limit}s) exceeded limit ({time_limit_sec}s)"
                    else:
                        metadata["timeout_message"] = f"User code subprocess timeout exceeded limit ({time_limit_sec}s)"
                    
                    return False, metadata

                if run_metadata.get("memory_limit_exceeded", False) is True:
                    metadata["user_code_timeout"] = True
                    metadata["timeout_source"] = "user_code"
                    metadata["error"] = "MemoryLimitExceeded"
                    metadata["error_type"] = "MemoryLimitExceeded"
                    return False, metadata
                
                # Check if solution ran successfully
                if run_result.returncode != 0:
                    # Don't retry on runtime errors (not timeout-related)
                    return False, metadata
                
                # Write solution output
                # print("input: ", input_data[:100])
                # print("solution output: ", run_result.stdout[:100])
                # print("expected answer: ", expected_output[:100])
                # raise ValueError
                output_file.write_text(run_result.stdout, encoding='utf-8')
                # Store stdout in metadata for potential use in first error tracking
                # metadata["stdout"] = run_result.stdout
                
                # Run checker: checker input.txt answer.txt output.txt (special checker)
                try:
                    checker_result = subprocess.run(
                        [str(checker_exe), str(input_file), str(answer_file), str(output_file)],
                        capture_output=True,
                        text=True,
                        timeout=checker_timeout,
                    )
                    # print("full results: ", checker_result)
                    # Checker returns 0 for AC, non-zero for WA
                    is_correct = checker_result.returncode == 0 and checker_result.stdout.strip() == "1"
                    
                    metadata["checker_returncode"] = checker_result.returncode
                    metadata["checker_stdout"] = checker_result.stdout
                    metadata["checker_stderr"] = checker_result.stderr
                    
                    if not is_correct:
                        metadata["checker_message"] = checker_result.stdout.strip() or checker_result.stderr.strip()
                    
                    return is_correct, metadata
                    
                except subprocess.TimeoutExpired as e:
                    metadata["checker_timeout"] = True
                    metadata["timeout_source"] = "checker"
                    metadata["timeout_message"] = f"Checker subprocess timeout ({checker_timeout}s) exceeded"
                    if attempt < max_retries:
                        # Retry on checker timeout
                        continue
                    return False, metadata
                except Exception as e:
                    metadata["checker_exception"] = str(e)
                    metadata["checker_exception_type"] = type(e).__name__
                    if attempt < max_retries:
                        # Retry on checker exception
                        continue
                    return False, metadata
                    
        except subprocess.TimeoutExpired as e:
            # This should not happen for the outer try, but handle it just in case
            metadata["outer_timeout"] = True
            metadata["timeout_source"] = "unknown"
            metadata["timeout_message"] = f"Outer subprocess timeout: {str(e)}"
            if attempt < max_retries:
                continue
            return False, metadata
        except Exception as e:
            metadata["outer_exception"] = str(e)
            metadata["outer_exception_type"] = type(e).__name__
            if attempt < max_retries:
                continue
            return False, metadata
    
    # If we exhausted all retries
    metadata["max_retries_exceeded"] = True
    return False, metadata


def run_cpp_tests_with_checker(
    solution: str,
    data_dir: str,
    problem_id: str,
    compile_timeout: int = 30,
) -> Tuple[List[bool], Dict[str, Any]]:
    """Compile and run C++ code with test cases using checker.
    
    Args:
        solution: C++ source code
        data_dir: Root data directory
        problem_id: Problem ID (subdirectory name)
        compile_timeout: Compilation timeout in seconds
        
    Returns:
        Tuple of (results list, metadata dict)
    """
    results = []
    metadata = {}
    
    # Load configuration
    try:
        config = load_problem_config(data_dir, problem_id)
    except Exception as e:
        metadata["error"] = "Config Load Error"
        metadata["error_message"] = str(e)
        return [], metadata
    
    # Load test cases
    try:
        inputs, outputs = load_test_cases(data_dir, problem_id, config)
    except Exception as e:
        metadata["error"] = "Test Cases Load Error"
        metadata["error_message"] = str(e)
        return [], metadata
    
    if not inputs:
        metadata["error"] = "No Test Cases"
        metadata["error_message"] = "No test cases found"
        return [], metadata
    
    # Get limits from config
    time_limit_sec = config.get('time_limit_sec', 2.0)
    memory_limit_mb = config.get('memory_limit_mb', 512)
    checker_file = config.get('checker')
    
    problem_dir = Path(data_dir) / problem_id
    
    # print("inputs", inputs[0])
    # print("outputs", outputs[0])
    # Create temporary directory for compilation
    with tempfile.TemporaryDirectory() as tmpdir:
        cpp_file = Path(tmpdir) / "solution.cpp"
        exe_file = Path(tmpdir) / "solution"
        
        # Preprocess C++ code to fix common issues (e.g., replace bits/stdc++.h)
        processed_solution = preprocess_cpp_code(solution)
        
        # Write C++ code to file
        cpp_file.write_text(processed_solution, encoding="utf-8")
        
        # Compile solution
        compile_success, compile_metadata = compile_cpp(cpp_file, exe_file, compile_timeout)
        metadata.update(compile_metadata)
        
        if not compile_success:
            return [False] * len(inputs), metadata
        
        # Compile checker if provided
        checker_exe = None
        if checker_file:
            checker_source = problem_dir / checker_file
            if checker_source.exists():
                checker_exe = Path(tmpdir) / "checker"
                checker_success, checker_compile_metadata = compile_checker(
                    checker_source, checker_exe, compile_timeout
                )
                metadata.update(checker_compile_metadata)
                if not checker_success:
                    metadata["checker_compile_failed"] = True
                    checker_exe = None
        
        # Track first error for adding input/output/answer to metadata
        first_error_idx = None
        first_error_input = None
        first_error_output = None
        first_error_answer = None
        
        # Run tests
        for idx, (input_data, expected_output) in enumerate(zip(inputs, outputs)):
            test_metadata = {}
            
            try:
                if checker_exe:
                    # Use checker for verification
                    is_correct, check_metadata = run_with_checker(
                        exe_file,
                        checker_exe,
                        input_data,
                        expected_output,
                        time_limit_sec,
                        memory_limit_mb,
                    )
                    # print("run_with_checker", is_correct, check_metadata)
                    test_metadata.update(check_metadata)
                    results.append(is_correct)
                    
                    if not is_correct and test_metadata.get("error_type", None) is None: 
                        test_metadata["error"] = True
                        test_metadata["error_type"] = "Wrong Answer (Checker)"
                    
                    # Track first error for metadata
                    if not is_correct and first_error_idx is None:
                        first_error_idx = idx
                        first_error_input = input_data[:1000] if input_data else None
                        first_error_answer = expected_output[:1000] if expected_output else None
                        # Try to get actual output from check_metadata if available
                        first_error_output = check_metadata.get("stdout", "")[:1000] if check_metadata.get("stdout") else None
                else:
                    # Fallback to direct output comparison
                    run_result, run_metadata = run_with_limits(
                        exe_file,
                        input_data,
                        time_limit_sec,
                        memory_limit_mb,
                    )
                    
                    test_metadata.update(run_metadata)
                    
                    if run_result.returncode != 0:
                        results.append(False)
                        test_metadata["error"] = True
                        
                        if "time_limit_exceeded" in run_metadata or "timeout" in run_metadata:
                            test_metadata["error_type"] = "Time Limit Exceeded"
                        elif "memory_limit_exceeded" in run_metadata:
                            test_metadata["error_type"] = "Memory Limit Exceeded"
                        else:
                            test_metadata["error_type"] = "Runtime Error"
                        
                        # Track first error for metadata
                        if first_error_idx is None:
                            first_error_idx = idx
                            first_error_input = input_data[:1000] if input_data else None
                            first_error_answer = expected_output[:1000] if expected_output else None
                            first_error_output = run_result.stdout[:1000] if run_result.stdout else None
                    else:
                        # Compare output directly
                        actual_output = run_result.stdout
                        is_correct = compare_output(actual_output, expected_output)
                        results.append(is_correct)
                        
                        if not is_correct:
                            test_metadata["output_mismatch"] = True
                            # test_metadata["actual_output"] = actual_output
                            # test_metadata["expected_output"] = expected_output
                            
                            # Track first error for metadata
                            if first_error_idx is None:
                                first_error_idx = idx
                                first_error_input = input_data[:1000] if input_data else None
                                first_error_output = actual_output[:1000] if actual_output else None
                                first_error_answer = expected_output[:1000] if expected_output else None
                
                # Store test metadata
                for key, value in test_metadata.items():
                    metadata[f"test_{idx}_{key}"] = value
                
            except Exception as e:
                results.append(False)
                metadata[f"test_{idx}_exception"] = str(e)
                metadata[f"test_{idx}_exception_type"] = type(e).__name__
                if "error_message" not in metadata:
                    metadata["error_message"] = f"Exception in test {idx}: {str(e)}"
                
                # Track first error for metadata
                if first_error_idx is None:
                    first_error_idx = idx
                    first_error_input = input_data[:1000] if input_data else None
                    first_error_answer = expected_output[:1000] if expected_output else None
        
        # Add first error input/output/answer to metadata if an error was found
        if first_error_idx is not None:
            metadata["first_error_input"] = first_error_input[:1000] if first_error_input else None
            metadata["first_error_output"] = first_error_output[:1000] if first_error_output else None 
            metadata["first_error_answer"] = first_error_answer[:1000] if first_error_answer else None
    
    return results, metadata


if __name__ == "__main__":
    # Sample test for problem 1983A
    import sys
    
    # Sample solution (you can replace this with actual solution)
    sample_solution = """
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>

using namespace std;

int main() {
    int t;
    cin >> t;
    for (int t_case = 0; t_case < t; ++t_case) {
        int n, k;
        cin >> n >> k;
        vector<int> a(n);
        for (int i = 0; i < n; ++i) {
            cin >> a[i];
        }
        sort(a.begin(), a.end());
        map<int, int> freq;
        for (int x : a) {
            freq[x]++;
        }
        vector<int> S;
        for (auto it = freq.begin(); it != freq.end(); ++it) {
            S.push_back(it->first);
        }
        vector<vector<int>> runs;
        if (!S.empty()) {
            vector<int> current_run;
            current_run.push_back(S[0]);
            for (size_t i = 1; i < S.size(); ++i) {
                if (S[i] == S[i-1] + 1) {
                    current_run.push_back(S[i]);
                } else {
                    runs.push_back(current_run);
                    current_run.clear();
                    current_run.push_back(S[i]);
                }
            }
            runs.push_back(current_run);
        }
        int max_cards = 0;
        for (auto& run : runs) {
            int m = run.size();
            vector<int> counts;
            for (int x : run) {
                counts.push_back(freq[x]);
            }
            int sum_total = accumulate(counts.begin(), counts.end(), 0);
            if (m <= k) {
                if (sum_total > max_cards) {
                    max_cards = sum_total;
                }
            } else {
                int current_sum = 0;
                for (int i = 0; i < k; ++i) {
                    current_sum += counts[i];
                }
                int max_k_sum = current_sum;
                for (size_t i = k; i < counts.size(); ++i) {
                    current_sum += counts[i] - counts[i - k];
                    if (current_sum > max_k_sum) {
                        max_k_sum = current_sum;
                    }
                }
                if (max_k_sum > max_cards) {
                    max_cards = max_k_sum;
                }
            }
        }
        cout << max_cards << endl;
    }
    return 0;
}
"""
    
    # Test with 1983A
    data_dir = "path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data"
    problem_id = "2025C"
    
    print(f"Testing problem {problem_id}...")
    print(f"Data directory: {data_dir}")
    
    try:
        results, metadata = run_cpp_tests_with_checker(
            solution=sample_solution,
            data_dir=data_dir,
            problem_id=problem_id,
        )
        
        passed = sum(1 for r in results if r is True)
        total = len(results)
        
        print(f"\nResults: {passed}/{total} test cases passed")
        
        if metadata.get("error"):
            print(f"Error: {metadata.get('error_message')}")
        
        # Print summary
        if passed == total:
            print("✅ All test cases passed!")
        else:
            print(f"❌ {total - passed} test cases failed")
            # Print first few failures
            for i, result in enumerate(results):
                if not result:
                    print(f"\nTest case {i+1} failed:")
                    for key, value in metadata.items():
                        if key.startswith(f"test_{i}_"):
                            print(f"  {key}: {value}")
                    if i >= 2:  # Only show first 3 failures
                        print(f"  ... and {total - passed - 3} more failures")
                        break
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

