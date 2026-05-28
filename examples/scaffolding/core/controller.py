# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.controller

import copy
import logging
from abc import ABC
from collections.abc import Mapping
from enum import Enum
from typing import Any

import torch

from .math_utils import get_digit_majority_vote_result
from .task import ChatTask, GenerationTask, Task

logger = logging.getLogger(__name__)


class Controller(ABC):
    task_collections: dict = {}

    def __init__(self):
        self.task_collections = {}

    def clone(self):
        return copy.deepcopy(self)

    def generate(self, prompt: str, **kwargs):
        task = GenerationTask.create_from_prompt(prompt)

        yield from self.process([task], **kwargs)

        return task.create_scaffolding_output()

    def process(self, tasks: list[Task], **kwargs):
        raise NotImplementedError


class ParallelProcess:
    def __init__(
        self,
        controllers: list[Controller],
        tasks_list: list[list[Task]],
        kwargs_list: list[Mapping[str, Any]],
    ):
        self.sub_gens = []
        for controller, tasks, kwargs in zip(controllers, tasks_list, kwargs_list):
            gen = controller.process(tasks, **kwargs)
            self.sub_gens.append(gen)


# Controller runs multiple generation tasks.
class NativeGenerationController(Controller):
    class WorkerTag(Enum):
        GENERATION = "generation"

    def __init__(self, sampling_params: dict = None, streaming: bool = False):
        super().__init__()
        if sampling_params is None:
            sampling_params = {}
        for key, value in list(sampling_params.items()):
            if key not in GenerationTask.__annotations__:
                logger.warning(f"{key} is not a supported field for GenerationTask")
                sampling_params.pop(key)
        self.sampling_params = sampling_params
        self.streaming = streaming

    # [GenerationTask] -> [GenerationTask] | [ChatTask] -> [ChatTask]
    def process(self, tasks: list[Task], **kwargs):
        for task in tasks:
            task.worker_tag = self.WorkerTag.GENERATION
            for key, value in self.sampling_params.items():
                if getattr(task, key) is None:
                    setattr(task, key, value)

            task.streaming_output_flag = self.streaming

        yield tasks


class NativeChatController(NativeGenerationController):
    def __init__(self, sampling_params: dict = None, streaming: bool = False):
        super().__init__(sampling_params, streaming)

    def process(self, tasks: list[Task], **kwargs):
        chat_tasks = [ChatTask.create_from_prompt(task.input_str) for task in tasks]
        yield from super().process(chat_tasks, **kwargs)


class NativeRewardController(Controller):
    def __init__(self):
        self.scores = None

    class WorkerTag(Enum):
        REWARD = "reward"

    def process(self, tasks: list[Task], **kwargs):
        task = GenerationTask()
        for task in tasks:
            task.worker_tag = self.WorkerTag.REWARD

        yield tasks


class MajorityVoteController(Controller):
    def __init__(self, generation_controller: Controller, default_sample_num: int = 1):
        super().__init__()
        self.generation_controller = generation_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        generation_controller = self.generation_controller.clone()
        return MajorityVoteController(generation_controller, self.default_sample_num)

    def process(
        self,
        tasks: list[Task],
        sample_num: int = 1,
        generation_kwargs: dict = {},
        majority_vote_kwargs: dict = {},
    ):
        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [
            self.generation_controller.clone() for _ in range(sample_num)
        ]
        tasks_list = [copy.deepcopy(tasks) for _ in range(sample_num)]
        generation_kwargs_list = [
            copy.deepcopy(generation_kwargs) for _ in range(sample_num)
        ]

        yield ParallelProcess(
            generation_controllers, tasks_list, generation_kwargs_list
        )

        majority_index, majority_answer = self.majority_vote(
            tasks_list, **majority_vote_kwargs
        )

        assert isinstance(majority_answer, str), "majority_vote failed"
        tasks[0].result = tasks_list[majority_index][0].result

    def majority_vote(
        self, candidates_tasks: list[list[Task]], **kwargs
    ) -> tuple[int, str]:
        candidates = [tasks[0].output_str for tasks in candidates_tasks]
        return get_digit_majority_vote_result(candidates)


class BestOfNController(Controller):
    def __init__(
        self,
        generation_controller: Controller,
        reward_controller: Controller,
        default_sample_num: int = 4,
    ):
        super().__init__()
        self.generation_controller = generation_controller
        self.reward_controller = reward_controller
        self.default_sample_num = default_sample_num

    def clone(self):
        generation_controller = self.generation_controller.clone()
        reward_controller = self.reward_controller.clone()
        return BestOfNController(
            generation_controller, reward_controller, self.default_sample_num
        )

    def process(
        self,
        tasks: list[Task],
        sample_num: int = 4,
        generation_kwargs: dict = {},
        reward_kwargs: dict = {},
        select_best_kwargs: dict = {},
    ):
        assert len(tasks) == 1, "BestOfNController only supports one task"
        task = tasks[0]

        sample_num = max(sample_num, self.default_sample_num)
        generation_controllers = [self.generation_controller for _ in range(sample_num)]
        generation_kwargs_list = [generation_kwargs for _ in range(sample_num)]
        generation_tasks = [copy.deepcopy(task) for _ in range(sample_num)]

        yield ParallelProcess(
            generation_controllers,
            [[t] for t in generation_tasks],
            generation_kwargs_list,
        )

        yield from self.reward_controller.process(generation_tasks, **reward_kwargs)

        assert self.reward_controller.scores is not None
        reward_values = self.reward_controller.scores

        for i, gen_task, reward_value in zip(
            range(sample_num), generation_tasks, reward_values
        ):
            logger.info(f"[output {i}, score {reward_value}]:\n{gen_task.output_str}")

        best_task, best_idx = self.select_best(
            generation_tasks, reward_values, **select_best_kwargs
        )
        task.result = best_task.result

    def select_best(self, tasks: list[Task], reward_values, **kwargs) -> Task:
        max_index = torch.argmax(torch.tensor(reward_values)).item()
        return tasks[max_index], max_index
