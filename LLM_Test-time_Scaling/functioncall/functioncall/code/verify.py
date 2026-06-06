import json
import os
import random
from collections import defaultdict
from datetime import datetime

from functioncall.base.call import Language, batch_function_call, get_runtime_name
from functioncall.base.utils import construct_uid, load_jsonl, logger

SINGLE_CASE_EXEC_TIMEOUT = 6
TEST_CASE_BATCH_SIZE = 1
FUNCTIONCALL_TIMEOUT = 100


def round_up_memory(memory):
    if memory <= 0:
        return 0
    rounded = ((memory + 255) // 256) * 256
    return 0 if rounded > 1024 else rounded


def construct_testcases(
    inputs: list, outputs: list, index: tuple, remote: bool = False, is_ut: bool = False
) -> dict:
    result = []
    if is_ut:
        return result

    for i in range(*index):
        input_, output_ = inputs[i].strip(), outputs[i].strip()
        if not remote:
            result.append({"input": input_, "expectedOutput": output_})
            continue

        oss_basepath = os.getenv("REAL_OSS_TESTCASE_PATH", "")
        if not oss_basepath:
            raise FileNotFoundError(
                "REAL_OSS_TESTCASE_PATH not set. Cannot use FAAS code reward."
            )
        input_url = (
            input_ if input_.startswith("http") else os.path.join(oss_basepath, input_)
        )
        output_url = (
            output_
            if output_.startswith("http")
            else os.path.join(oss_basepath, output_)
        )

        result.append({"input": input_url, "expectedOutput": output_url})
    return result


def load_problems_with_testcase_batch(
    id2info, query_ids, generateds, timeout_for_testcase, test_case_batch_size
):
    problem_list = []
    for idx, query_id in enumerate(query_ids):
        problem = id2info[query_id]
        # parse one problem
        language = problem.get("language", "PYTHON").upper()
        timeout = min(
            100, max(0.1, float(problem.get("timeout", timeout_for_testcase)) * 1.5)
        )  # [0.1, 100] s
        # timeout = 10
        memory = round_up_memory(problem.get("memory", 0))
        input_output = json.loads(problem["input_output"])
        fn_name = input_output.get("fn_name", "")
        remote = input_output.get("remote", False)
        inputs = input_output.get("inputs", [])
        outputs = input_output.get("outputs", [])

        assert len(inputs) == len(
            outputs
        ), f"Inputs({len(inputs)}) and outputs({len(outputs)}) mismatch for {query_id}"

        assert (
            language in Language.__members__
        ), f"{language} is not a valid Language name"

        is_ut = len(inputs) == 0

        # isFastFail means the function call returns immediately as soon as any testcase fails.
        isFastFail = True
        # create batches for testcases
        case_size = 1 if is_ut else len(inputs)
        test_case_batch_size = min(max(1, test_case_batch_size), case_size)

        for batch_idx in range(0, case_size, test_case_batch_size):
            end_idx = min(case_size, batch_idx + test_case_batch_size)
            testcases = construct_testcases(
                inputs, outputs, (batch_idx, end_idx), remote, is_ut
            )
            runtime = get_runtime_name("", language)
            sub_problem = {
                "uid": construct_uid(query_id, batch_idx, end_idx),
                "language": language,
                "runtime": runtime,
                "code": generateds[idx],
                "entryFunction": fn_name,
                "isFastFail": isFastFail,
                "isRemote": remote,
                "testcases": testcases,
                "timeout": timeout,
                "memory": memory,
                "query_index": idx,
            }
            if language == "CPP":
                sub_problem["runtime"] = "cpp-bugfix"
                sub_problem["extraInfo"] = {
                    "compileTimeout": 100   
                }
            problem_list.append(sub_problem)

    return problem_list


def code_verify(
    id2info,
    generateds,
    query_ids,
    timeout=FUNCTIONCALL_TIMEOUT,
    timeout_for_testcase=SINGLE_CASE_EXEC_TIMEOUT,
    test_case_batch_size=TEST_CASE_BATCH_SIZE,
):
    assert len(generateds) == len(query_ids), (
        len(generateds),
        len(query_ids),
    )
    payload_list = []

    payload_list = load_problems_with_testcase_batch(
        id2info,
        query_ids,
        generateds,
        timeout_for_testcase,
        test_case_batch_size,
    )

    for num_attempts in range(4):
        logger.info(
            f"code_verify start, request count: {len(payload_list)}, query size: {len(query_ids)}, query_id_0: {query_ids[0]}"
        )
        rsp_list = batch_function_call(payload_list, "code", timeout)

        results = [1] * len(query_ids) if len(rsp_list) else [0] * len(query_ids)
        rerun = False
        if len(rsp_list):
            time_elapsed = rsp_list[0]["time_elapsed"]
        for idx, rsp in enumerate(rsp_list):
            query_index = payload_list[idx]["query_index"]
            query_id = query_ids[query_index]

            value = 0
            if rsp and rsp.get("success", False):
                value = 1
            else:
                if "TimeoutExpired" in rsp["results"][0].get("stderr", "") and time_elapsed < 9:
                    logger.info(
                        f"code_verify find run wrong timeout expired for problem {query_id}: {rsp}"
                    )
                    if num_attempts < 3:
                        rerun = True
                        break
                if "Command '['g++', '-O2', '-o', 'test', 'test.cpp']' timed out" in rsp["results"][0].get("stderr", ""):
                    logger.info(
                        f"code_verify find compile timeout for problem {query_id}: {rsp}"
                    )
                logger.debug(
                    f'Functioncall code verify not passed, uid: {rsp.get("uid")}, query id: {query_id}, results: {rsp}'
                )

            results[query_index] = results[query_index] and value

        if rerun is True:
            continue

        logger.info(
            f"code_verify finished, request count: {len(payload_list)}, query count: {len(query_ids)}, result count: {len(results)}"
        )
        return results
    raise ValueError

if __name__ == "__main__":
    data_list = load_jsonl(
        "path-to-test-case/test_success_dataset.jsonl"
    )
    id2info = defaultdict(dict)
    for item in data_list:
        id2info[item["query_id"]] = item

    def create_test_params(count=10):
        query_ids = []
        generateds = []
        cnt = 0

        for d in data_list:
            if cnt >= count:
                break
            if d["query_id"] not in id2info:
                continue
            query_ids.append(d["query_id"])
            generateds.extend(d["solutions"])
            cnt += 1

        return generateds, query_ids

    generateds, query_ids = create_test_params(100)
    scale = 1
    print(f"generateds:, query_ids:{query_ids}")
    result = code_verify(id2info, generateds * scale, query_ids * scale)
    print(result)
