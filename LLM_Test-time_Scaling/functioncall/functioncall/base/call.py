import asyncio
import logging
import os
import random
import time
from enum import Enum
from statistics import median
from typing import Any, Dict

import aiohttp

try:
    from realhf.base import constants, logging
except Exception:
    import logging

    constants = None

logger = logging.getLogger("function call")

FUNCTIONCALL_SERVICE_DOMAIN = os.getenv(
    "FUNCTIONCALL_SERVICE_DOMAIN",
    "",
)


def check_payload(payload):
    if not payload:
        return False, {
            "uid": payload.get("uid", ""),
            "success": False,
            "results": [
                {
                    "success": False,
                    "reason": "Empty payload",
                    "errorType": "UnknownError",
                }
            ],
        }
    if not payload.get("code"):
        return False, {
            "uid": payload.get("uid", ""),
            "success": False,
            "results": [
                {"success": False, "reason": "Empty code", "errorType": "UnknownError"}
            ],
        }
    return True, {}


class Language(Enum):
    PYTHON = 0
    JAVA = 1
    CPP = 2
    C = 3
    MATH = 4
    SQL = 5
    GO = 6
    NODEJS = 7
    CSHARP = 8
    TYPESCRIPT = 9
    JAVASCRIPT = 10

    def __str__(self):
        return f"{self.name.lower()}"


def calculate_percentile(elapsed_times, percentile):
    sorted_times = sorted(elapsed_times)
    index = int(len(sorted_times) * (percentile / 100))
    return sorted_times[min(index, len(sorted_times) - 1)]


def has_system_error(response_json):
    for result in response_json.get("results", []):
        if result.get("errorType", "") == "SystemError":
            return True, result
    return False, None


async def async_invoke_function(
    session: aiohttp.ClientSession,
    url: str,
    timeout: aiohttp.ClientTimeout,
    payload: Dict[str, Any] = None,
    max_retries: int = 2,
    initial_retry_interval: float = 0.5,
    max_retry_interval: float = 10.0,
):

    retries = 0
    while retries < max_retries:
        try:
            async with session.post(
                url,
                json=payload,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(
                        f"HTTP Error {response.status}: {text} : {response.headers}"
                    )

                try:
                    response_json = await response.json()

                    exist, err_info = has_system_error(response_json)
                    if exist:
                        raise Exception(
                            f'SystemError detected, uid: {response_json.get("uid")}, err: {err_info}'
                        )

                    return response_json
                except aiohttp.ContentTypeError as e:
                    raise Exception("Invalid JSON response") from e

        except asyncio.TimeoutError as e:
            logger.warning(
                f'Request timeout after {timeout}s, uid: {payload.get("uid")}, URL: {url}'
            )
            return {
                "uid": payload.get("uid", ""),
                "success": False,
                "results": [
                    {
                        "success": False,
                        "reason": "Function call timed out.",
                        "errorType": "UnknownError",
                    }
                ],
            }

        except Exception as e:
            logger.error(
                f"Async invocation failed on attempt {retries + 1}:{str(e)}, uid: {payload.get('uid')}, URL: {url}"
            )

        retries += 1
        if retries >= max_retries:
            return {
                "uid": payload.get("uid", ""),
                "success": False,
                "results": [
                    {
                        "success": False,
                        "reason": "Function call exceed max retries.",
                        "errorType": "UnknownError",
                    }
                ],
            }

        sleep_time = min(
            initial_retry_interval * (2**retries) + random.uniform(0, 5),
            max_retry_interval,
        )
        await asyncio.sleep(sleep_time)


async def batch_function_call_async(payload_list, url, timeout, concurrency=1500):
    connector = aiohttp.TCPConnector(
        limit=concurrency,
        ttl_dns_cache=300,  # DNS cache
        keepalive_timeout=80,  # keepalive_timeout need to be smaller than the middle link idle-timeout
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_task(payload):
            ok, err_rsp = check_payload(payload)
            if not ok:
                return err_rsp, 0
            async with semaphore:
                st = time.monotonic()
                result = await async_invoke_function(session, url, timeout, payload)
                return result, time.monotonic() - st

        tasks = [limited_task(payload) for payload in payload_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = results if results else []
        data_list = []
        elapsed_times = []
        max_elapsed = -1
        max_elapsed_uid = ""
        for data, elapsed in results:
            if elapsed > max_elapsed:
                max_elapsed = elapsed
                max_elapsed_uid = data.get("uid")
            data_list.append(data)
            elapsed_times.append(elapsed)

        if len(elapsed_times) > 0:
            p50 = median(elapsed_times)
            p90 = calculate_percentile(elapsed_times, 90)
            p99 = calculate_percentile(elapsed_times, 99)
            logger.info(
                f"Longest functioncall took {max_elapsed:.4f} seconds, timeout: {timeout}, uid: {max_elapsed_uid}, Active connections: {len(connector._conns)}, p50: {p50}, p90: {p90}, p99: {p99}"
            )

        return data_list


def get_runtime_name(runtime, language):
    if runtime:
        return runtime
    else:
        return str(language).lower() + "-default"


def caculate_concurrency():
    # use 5000 cpu cores for one exp by default
    concurrency_for_one_exp = 5000
    try:
        dp = constants.parallelism_group_size()
    except Exception as e:
        dp = 16
    return concurrency_for_one_exp // dp


def batch_function_call(payload_list, task_type, timeout):
    start_time = time.time()
    url = f"{FUNCTIONCALL_SERVICE_DOMAIN}/apis/functioncalls"

    concurrency = caculate_concurrency()
    logger.info(
        f"Batch function call start, task type: {task_type}, request count: {len(payload_list)}, time: {time.ctime(start_time)} ms, concurrency: {concurrency}"
    )
    result = asyncio.run(
        batch_function_call_async(payload_list, url, timeout, concurrency=concurrency)
    )
    execution_time = time.time() - start_time
    logger.info(
        f"Batch function call done, task type: {task_type}, batch size: {len(payload_list)}, cost: {execution_time * 1000:.0f} ms"
    )
    result[0]["time_elapsed"] = execution_time
    return result
