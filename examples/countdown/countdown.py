import argparse
import itertools
import pathlib
import random
import re
import sys

from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, str(pathlib.Path().resolve()))


def get_existing_problems():
    existing_problems = set()
    # All problems from the JS file, organized by sets of 10
    conditions = [
        # Set 1
        [1, 1, 5, 5],
        [1, 3, 6, 7],
        [1, 5, 5, 6],
        [7, 7, 11, 12],
        [6, 8, 12, 12],
        [3, 4, 8, 12],
        [7, 8, 8, 9],
        [2, 3, 5, 10],
        [5, 5, 7, 10],
        [1, 2, 7, 7],
        # Set 2
        [1, 6, 6, 11],
        [7, 9, 13, 13],
        [1, 6, 8, 13],
        [2, 3, 10, 13],
        [4, 5, 6, 10],
        [2, 6, 8, 13],
        [3, 7, 7, 9],
        [2, 5, 8, 10],
        [1, 5, 5, 10],
        [3, 5, 10, 10],
        # Set 3
        [5, 5, 12, 12],
        [1, 2, 3, 8],
        [6, 7, 12, 12],
        [4, 10, 12, 12],
        [3, 9, 10, 12],
        [2, 4, 6, 13],
        [1, 3, 7, 12],
        [1, 5, 6, 11],
        [6, 8, 11, 12],
        [5, 10, 10, 11],
        # Set 4
        [1, 5, 7, 11],
        [1, 8, 9, 12],
        [3, 3, 9, 9],
        [3, 4, 8, 10],
        [2, 4, 6, 9],
        [6, 6, 11, 12],
        [4, 4, 5, 10],
        [7, 8, 8, 11],
        [3, 3, 9, 11],
        [4, 8, 8, 11],
        # Set 5
        [6, 6, 6, 6],
        [11, 13, 13, 13],
        [3, 3, 11, 12],
        [5, 10, 13, 13],
        [1, 4, 5, 11],
        [7, 9, 9, 13],
        [1, 1, 7, 10],
        [6, 9, 9, 11],
        [7, 10, 10, 12],
        [2, 2, 10, 11],
        # Set 6
        [2, 2, 3, 12],
        [3, 8, 12, 12],
        [2, 8, 9, 12],
        [1, 11, 12, 13],
        [6, 6, 8, 12],
        [4, 4, 8, 11],
        [6, 7, 9, 12],
        [1, 4, 5, 8],
        [3, 5, 9, 9],
        [1, 6, 11, 13],
        # Set 7
        [1, 1, 4, 6],
        [1, 4, 7, 13],
        [1, 4, 4, 4],
        [4, 7, 7, 8],
        [2, 3, 9, 10],
        [4, 7, 12, 12],
        [3, 3, 6, 12],
        [5, 5, 8, 13],
        [2, 2, 7, 7],
        [2, 4, 7, 12],
        # Set 8
        [1, 1, 11, 11],
        [1, 2, 4, 4],
        [4, 4, 5, 6],
        [1, 5, 10, 12],
        [6, 6, 8, 9],
        [1, 2, 7, 11],
        [2, 2, 3, 11],
        [3, 6, 8, 13],
        [1, 2, 8, 10],
        [2, 2, 7, 10],
        # Set 9
        [1, 4, 7, 12],
        [1, 7, 8, 10],
        [5, 7, 13, 13],
        [3, 6, 12, 12],
        [1, 3, 6, 13],
        [2, 7, 9, 13],
        [2, 2, 5, 12],
        [3, 9, 10, 13],
        [4, 7, 8, 12],
        [2, 7, 7, 10],
        # Set 10
        [1, 1, 2, 6],
        [10, 11, 11, 12],
        [9, 10, 10, 13],
        [5, 6, 8, 8],
        [2, 2, 9, 11],
        [5, 8, 8, 9],
        [2, 4, 5, 9],
        [5, 5, 8, 10],
        [3, 5, 7, 11],
        [1, 3, 9, 10],
        # Set 11
        [1, 2, 2, 6],
        [1, 8, 8, 12],
        [1, 8, 10, 12],
        [1, 3, 6, 9],
        [4, 4, 4, 7],
        [3, 4, 8, 11],
        [3, 5, 7, 10],
        [1, 7, 10, 13],
        [2, 8, 10, 12],
        [2, 3, 13, 13],
        # Set 12
        [2, 2, 11, 13],
        [1, 4, 6, 13],
        [1, 2, 5, 7],
        [1, 11, 11, 12],
        [1, 4, 12, 12],
        [1, 3, 3, 10],
        [3, 3, 6, 10],
        [7, 12, 12, 13],
        [2, 3, 7, 10],
        [3, 5, 8, 13],
        # Set 13
        [3, 3, 12, 12],
        [9, 9, 11, 13],
        [1, 3, 3, 7],
        [2, 3, 3, 7],
        [4, 5, 5, 9],
        [2, 2, 5, 11],
        [6, 6, 7, 10],
        [4, 4, 9, 11],
        [4, 7, 8, 11],
        [8, 9, 11, 11],
        # Set 14
        [1, 1, 2, 13],
        [1, 1, 5, 8],
        [2, 12, 12, 13],
        [3, 5, 6, 8],
        [4, 7, 8, 13],
        [6, 9, 9, 12],
        [3, 3, 6, 13],
        [8, 9, 9, 12],
        [2, 6, 6, 7],
        [5, 9, 10, 11],
        # Set 15
        [1, 6, 8, 9],
        [8, 9, 12, 13],
        [4, 8, 8, 12],
        [1, 5, 9, 10],
        [6, 7, 8, 10],
        [1, 6, 12, 13],
        [5, 5, 10, 10],
        [3, 5, 6, 11],
        [3, 5, 12, 12],
        [5, 6, 8, 13],
        # Set 16
        [2, 2, 12, 12],
        [5, 5, 7, 7],
        [7, 9, 11, 11],
        [2, 2, 3, 3],
        [4, 4, 8, 10],
        [5, 5, 6, 11],
        [3, 9, 13, 13],
        [2, 8, 10, 13],
        [2, 2, 6, 7],
        [2, 3, 7, 9],
        # Set 17
        [1, 2, 3, 4],
        [1, 1, 5, 6],
        [1, 4, 8, 11],
        [5, 6, 7, 8],
        [3, 3, 6, 11],
        [1, 5, 10, 13],
        [3, 5, 7, 9],
        [7, 8, 8, 12],
        [2, 6, 8, 9],
        [9, 11, 12, 13],
        # Set 18
        [4, 6, 13, 13],
        [1, 2, 2, 13],
        [1, 11, 12, 12],
        [3, 4, 7, 9],
        [2, 3, 6, 6],
        [5, 6, 7, 7],
        [3, 3, 3, 9],
        [3, 3, 3, 4],
        [8, 8, 8, 11],
        [3, 3, 7, 13],
        # Set 19
        [12, 12, 13, 13],
        [6, 8, 10, 12],
        [2, 6, 10, 10],
        [1, 2, 11, 13],
        [6, 8, 8, 10],
        [4, 5, 5, 8],
        [5, 6, 7, 13],
        [6, 7, 9, 9],
        [6, 10, 10, 13],
        [4, 4, 7, 7],
        # Set 20
        [4, 6, 11, 11],
        [8, 9, 12, 12],
        [1, 2, 3, 3],
        [1, 2, 7, 9],
        [1, 11, 13, 13],
        [5, 8, 9, 11],
        [2, 3, 5, 13],
        [2, 3, 6, 7],
        [7, 10, 11, 13],
        [1, 4, 5, 6],
    ]

    # Convert each problem to a sorted tuple and add to set
    for nums in conditions:
        sorted_nums = tuple(sorted(nums))
        existing_problems.add(sorted_nums)

    return existing_problems


SYSTEM = """Solve the problem step by step. Write your thoughts in <think> </think> tags.
The answer is a formula consisting of arithmetic operations (+, -, *, /) that results in the target number.

Write the final answer in <answer> </answer> tags.
Otherwise, the grader will not be able to parse your answer.

Example:
<think>thought process here</think>
<answer> (1 + 2) * 2 * 4 </answer>"""
ASSISTANT = "Answer: <think>Let's think step by step:\n"


def parse_solutions_words(result):
    result = result.strip()
    if "</final_answer>" not in result:
        print("warning, no answer found")
        return None
    try:
        answer = re.findall(r"<final_answer>(.*?)</final_answer>", result, re.DOTALL)[
            -1
        ]
    except Exception as e:
        print(f"warning, no answer found, {e}")
        answer = None
    # print(f"Result raw: {result}")
    # print(f"Answer raw: {answer}")
    return answer


def combine_nums(a, b):
    # Implicitly makes assumptions about the order of operations and valid operations
    a = int(a)
    b = int(b)
    possible = [[a + b, f"{a}+{b}={a + b}"], [a * b, f"{a}*{b}={a * b}"]]
    if a <= b:
        possible.append([b - a, f"{b}-{a}={b - a}"])
        if a != 0 and b % a == 0:
            possible.append([b // a, f"{b}/{a}={round(b // a, 0)}"])
    else:
        possible.append([a - b, f"{a}-{b}={a - b}"])
        if b != 0 and a % b == 0:
            possible.append([a // b, f"{a}/{b}={round(a // b, 0)}"])
    return possible


class CountDown:
    def __init__(
        self,
        max_target=25,
        start_size=[2, 3, 4],
        min_target=10,
        start_probs=[0.0, 0.4, 0.6],
        tokenizer_path: str = "Qwen/Qwen2.5-3B-Instruct",
    ):
        self.max_target = max_target
        self.min_target = min_target
        self.start_size = start_size
        self.start_probs = start_probs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.existing_problems = get_existing_problems()

    def is_duplicate(self, nums):
        return tuple(sorted(nums)) in self.existing_problems

    def generate(self, target):
        if target > self.max_target:
            raise ValueError("Target cannot be greater than max target")
        if target < self.min_target:
            raise ValueError("Target cannot be less than min target")

        found = False
        while not found:
            # nums in question can go up to max target
            start_size = random.choices(self.start_size, weights=self.start_probs)[0]
            nums = [random.randint(1, self.max_target - 1) for _ in range(start_size)]

            if self.is_duplicate(nums):
                continue

            solution = self.search(target, nums)
            if solution is not None:
                found = True
        return nums, solution

    def get_task(self, apply_chat_template=False, return_raw=False) -> dict[str, str]:
        target = random.randint(self.min_target, self.max_target)
        nums, solution = self.generate(target)

        query = (
            # "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
            # "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
            # "User: "
            f"Using the numbers {nums}, create an equation that equals {target}. "
            "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
            "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.\n"
            # "Assistant: Let me solve this step by step."
        )

        if return_raw:
            return {"query": query}

        messages = [
            {
                "role": "system",
                "content": SYSTEM,
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        if apply_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt += ASSISTANT
        else:
            """using a base model"""
            prompt = f"""{SYSTEM}\n\n{query}\n{ASSISTANT}"""
            messages[-1]["content"] = prompt
        self.current_task = {"query": prompt, "target": target, "numbers": nums}
        return self.current_task

    def search(self, target, nums, operations=[]):
        # Navigate the entire solution tree, implemented with DFS
        if len(nums) == 1:
            if nums[0] == target:
                return operations
            else:
                return None

        for i, j in itertools.combinations(range(len(nums)), 2):
            num1, num2 = nums[i], nums[j]
            remaining_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
            for result, operation in combine_nums(num1, num2):
                new_nums = remaining_nums + [result]
                new_operations = operations + [operation]
                solution = self.search(target, new_nums, new_operations)
                if solution is not None:
                    return solution
        return None


def create_countdown_datasets(
    seed=42,
    num_samples=500000,
    eval_size=1000,
    tokenizer_path="Qwen/Qwen2.5-3B-Instruct",
):
    random.seed(seed)
    countdown = CountDown(
        start_probs=[0.1, 0.4, 0.5],
        max_target=25,
        min_target=10,
        tokenizer_path=tokenizer_path,
    )

    train_data = []
    val_data = []
    test_data = []

    for _ in tqdm(range(num_samples), desc="Generating training data"):
        task = countdown.get_task(apply_chat_template=True)
        train_data.append(task)

    for _ in tqdm(range(eval_size), desc="Generating validation/test data"):
        task = countdown.get_task(apply_chat_template=True)
        val_data.append(task)

        task = countdown.get_task(apply_chat_template=True)
        test_data.append(task)

    return train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500000,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Number of validation/test samples to generate",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="The path or HF identifier of the tokenizer",
    )
    args = parser.parse_args()

    countdown = CountDown(tokenizer_path=args.tokenizer_path)
    task = countdown.get_task(apply_chat_template=True)
    print(task["query"])
    #     # get answer
    #     answer = """
    # Step 1: 1+2=3
    # Step 2: 3*3=9
    # Step 3: 9*3=27
    # Step 4: 27+3=30
    # """
    #     q="Find a sequence of arithmetic operations (+, -, *, /) that results in 14 using the numbers 2, 24, 12"
    #     answer = """
    #  Step 1: 24/2 = 12
    #  Step 2: 12 + 2 = 14
    # """
    #     print(countdown.verify_answer(14, q, answer))
    train_data, val_data, test_data = create_countdown_datasets(
        num_samples=args.num_samples,
        eval_size=args.eval_size,
        tokenizer_path=args.tokenizer_path,
    )
    print(len(train_data), len(val_data), len(test_data))
    # save to each to jsonl file
    import json

    with open("./data/countdown/qwen/train_e.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    with open("./data/countdown/qwen/valid_e.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    with open("./data/countdown/qwen/test_e.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
