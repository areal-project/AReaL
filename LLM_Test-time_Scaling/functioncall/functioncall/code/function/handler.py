import json

from testing_util import run_test


def handle(event, context):
    problem = event.get("problem", "")
    code = event.get("code", "")
    debug = event.get("debug", False)
    timeout = event.get("timeout", 6)

    if isinstance(problem, str):
        problem = json.loads(problem)

    # print(f"problem:{problem}, code:{code}, debug: {debug}\n")
    result, metadata = run_test(problem, test=code, debug=debug, timeout=timeout)
    return {"result": result, "metadata": metadata}


# handle(
#     {
#         "problem": {
#             "input_output": '{"inputs": ["1\\n", "2\\n"], "outputs": ["1\\n", "2\\n"]}'
#         },
#         "code": "s = input()\nprint(s)\n",
#         "debug": True,
#     },
#     None,
# )
