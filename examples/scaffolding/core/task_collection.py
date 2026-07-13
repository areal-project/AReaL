# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Vendored from tensorrt_llm.scaffolding.task_collection

import json
import time
from typing import Any

from .controller import ParallelProcess
from .task import ChatTask, GenerationTask, Task


class TaskCollection:
    def __init__(self):
        # reserved for future use
        pass

    def before_yield(self, tasks: list[Task]):
        pass

    def after_yield(self, tasks: list[Task]):
        pass

    @staticmethod
    def get_global_info() -> Any:
        pass


def with_task_collection(
    name: str, task_collection_cls: type[TaskCollection], **task_collection_kwargs
):
    def decorator(controller_cls: type):
        original_init = controller_cls.__init__
        original_process = controller_cls.process

        # add task collection to controller
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.task_collections[name] = task_collection_cls(**task_collection_kwargs)

        def new_process(self, tasks: list[Task], **kwargs):
            class TaskCollectionWrapper:
                def __init__(self, task_collection, gen):
                    self.task_collection = task_collection
                    self.gen = gen

                def __call__(self):
                    for obj in self.gen:
                        if isinstance(obj, ParallelProcess):
                            new_sub_gens = []
                            for sub_gen in obj.sub_gens:
                                new_sub_gen = TaskCollectionWrapper(
                                    self.task_collection, sub_gen
                                )
                                new_sub_gens.append(new_sub_gen)
                            obj.sub_gens = new_sub_gens

                            yield obj
                        else:  # obj is a list of tasks
                            self.task_collection.before_yield(obj)
                            yield obj
                            self.task_collection.after_yield(obj)

                def __iter__(self):
                    return self.__call__()

            original_gen = original_process(self, tasks, **kwargs)
            new_gen = TaskCollectionWrapper(self.task_collections[name], original_gen)
            return new_gen()

        controller_cls.__init__ = new_init
        controller_cls.process = new_process

        return controller_cls

    return decorator


class GenerationTokenCounter(TaskCollection):
    def __init__(self):
        super().__init__()
        self.generation_token_count = 0
        self.pre_worker_token_sum = 0

    def before_yield(self, tasks: list[Task]):
        self.pre_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask) or issubclass(
                type(task), GenerationTask
            ):
                if task.output_tokens:
                    self.pre_worker_token_sum += len(task.output_tokens)

    def after_yield(self, tasks: list[Task]):
        post_worker_token_sum = 0
        for task in tasks:
            if isinstance(task, GenerationTask) or issubclass(
                type(task), GenerationTask
            ):
                if task.output_tokens:
                    post_worker_token_sum += len(task.output_tokens)
        self.generation_token_count += post_worker_token_sum - self.pre_worker_token_sum


class ChatTokenCounter(TaskCollection):
    # prompt tokens, completion tokens
    statistics: dict[str, list[tuple[int, int]]] = {}

    def __init__(self, statistics_name: str):
        super().__init__()
        self.statistics_name = statistics_name
        if statistics_name not in ChatTokenCounter.statistics:
            ChatTokenCounter.statistics[statistics_name] = []

    def before_yield(self, tasks: list[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            task.enable_token_counting = True

    def after_yield(self, tasks: list[Task]):
        for task in tasks:
            if not isinstance(task, ChatTask):
                continue
            ChatTokenCounter.statistics[self.statistics_name].append(
                (task.prompt_tokens_num, task.completion_tokens_num)
            )

    def get_global_info() -> Any:
        return ChatTokenCounter.statistics


class TaskTimer(TaskCollection):
    statistics: dict[str, dict[type, list[float]]] = {}

    def __init__(self, statistics_name: str, task_types: list[type[Task]]):
        super().__init__()
        self.statistics_name = statistics_name
        self.task_types = task_types
        self.start_time_map = {}
        if statistics_name not in TaskTimer.statistics:
            TaskTimer.statistics[statistics_name] = {}
        for task_type in task_types:
            if task_type not in TaskTimer.statistics[statistics_name]:
                TaskTimer.statistics[statistics_name][task_type] = []

    def before_yield(self, tasks: list[Task]):
        for task in tasks:
            if type(task) not in self.task_types:
                continue

            self.start_time_map[id(task)] = time.time()

    def after_yield(self, tasks: list[Task]):
        for task in tasks:
            if type(task) not in self.task_types:
                continue

            end_time = time.time()
            TaskTimer.statistics[self.statistics_name][type(task)].append(
                end_time - self.start_time_map[id(task)]
            )
            del self.start_time_map[id(task)]

    def get_global_info() -> Any:
        return TaskTimer.statistics


class TaskMetricsCollector(TaskCollection):
    """Task profiler that captures tasks at yield points."""

    # Global statistics: controller_name -> List[task_info_dict]
    statistics: dict[str, list[dict[str, Any]]] = {}

    def __init__(
        self,
        controller_name: str,
        task_types: list[type[Task]] = None,
        enable_print: bool = True,
        capture_messages: bool = False,
    ):
        super().__init__()
        self.controller_name = controller_name
        self.task_types = task_types
        self.enable_print = enable_print
        self.capture_messages = capture_messages
        self.start_time_map: dict[int, float] = {}
        self.pre_message_count_map: dict[int, int] = {}

        if controller_name not in TaskMetricsCollector.statistics:
            TaskMetricsCollector.statistics[controller_name] = []

    def _should_process_task(self, task: Task) -> bool:
        if self.task_types is not None and type(task) not in self.task_types:
            return False
        return True

    def _is_task_already_profiled(self, task: Task) -> bool:
        return getattr(task, "_profiling_in_progress", False)

    def _mark_task_profiling_start(self, task: Task):
        task._profiling_in_progress = True

    def _mark_task_profiling_end(self, task: Task):
        task._profiling_in_progress = False

    def before_yield(self, tasks: list[Task]):
        for task in tasks:
            if not self._should_process_task(task):
                continue
            if self._is_task_already_profiled(task):
                continue

            self._mark_task_profiling_start(task)
            task_id = id(task)
            self.start_time_map[task_id] = time.time()

            if isinstance(task, ChatTask):
                task.enable_token_counting = True
                if self.capture_messages:
                    self.pre_message_count_map[task_id] = len(task.messages)

    def after_yield(self, tasks: list[Task]):
        for task in tasks:
            task_id = id(task)
            if task_id not in self.start_time_map:
                continue

            end_time = time.time()
            duration = end_time - self.start_time_map[task_id]
            del self.start_time_map[task_id]
            self._mark_task_profiling_end(task)

            task_info = {
                "controller": self.controller_name,
                "task_type": type(task).__name__,
                "duration_ms": duration * 1000,
                "timestamp": end_time,
            }

            if isinstance(task, ChatTask):
                task_info["prompt_tokens"] = getattr(task, "prompt_tokens_num", 0)
                task_info["completion_tokens"] = getattr(
                    task, "completion_tokens_num", 0
                )
                task_info["reasoning_tokens"] = getattr(task, "reasoning_tokens_num", 0)
                task_info["total_tokens"] = (
                    task_info["prompt_tokens"] + task_info["completion_tokens"]
                )
                task_info["finish_reason"] = getattr(task, "finish_reason", None)
                task_info["unique_id"] = getattr(task, "unique_id", None)
                task_info["sub_request_markers"] = getattr(
                    task, "sub_request_markers", []
                )
                task_info["perf_metrics"] = getattr(task, "perf_metrics", None)

                if self.capture_messages:
                    pre_message_count = self.pre_message_count_map.get(task_id, 0)
                    if task_id in self.pre_message_count_map:
                        del self.pre_message_count_map[task_id]

                    task_info["message_count_before"] = pre_message_count
                    task_info["message_count_after"] = len(task.messages)
                    task_info["messages"] = [
                        self._serialize_message(msg) for msg in task.messages
                    ]
                    if len(task.messages) > pre_message_count:
                        task_info["new_messages"] = [
                            self._serialize_message(msg)
                            for msg in task.messages[pre_message_count:]
                        ]
                    else:
                        task_info["new_messages"] = []

            TaskMetricsCollector.statistics[self.controller_name].append(task_info)

            if self.enable_print:
                self._print_task_info(task_info)

    def _serialize_message(self, message) -> dict[str, Any]:
        """Serialize a RoleMessage to a dictionary."""
        result = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        if hasattr(message, "reasoning") and message.reasoning is not None:
            result["reasoning"] = message.reasoning
        if (
            hasattr(message, "reasoning_content")
            and message.reasoning_content is not None
        ):
            result["reasoning_content"] = message.reasoning_content
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            result["tool_calls"] = [str(tc) for tc in message.tool_calls]
        return result

    def _print_task_info(self, task_info: dict[str, Any]):
        log_parts = [
            f"[{task_info['controller']}]",
            f"{task_info['task_type']}",
            f"duration={task_info['duration_ms']:.2f}ms",
        ]

        if "prompt_tokens" in task_info:
            log_parts.append(
                f"prompt={task_info['prompt_tokens']} "
                f"completion={task_info['completion_tokens']} "
                f"reasoning={task_info['reasoning_tokens']} "
                f"total={task_info['total_tokens']}"
            )

        if task_info.get("perf_metrics"):
            perf_str = ", ".join(
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in task_info["perf_metrics"].items()
            )
            log_parts.append(f"perf: {perf_str}")

        print(" | ".join(log_parts))

        if "new_messages" in task_info and task_info["new_messages"]:
            print(
                f"    Messages: {task_info['message_count_before']} -> {task_info['message_count_after']}"
            )
            print("    New Messages:")
            for msg in task_info["new_messages"]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content and len(content) > 200:
                    content = content[:200] + "..."
                print(f"      [{role}]: {content}")

    @staticmethod
    def _compute_stats(values: list[float]) -> dict[str, float]:
        """Compute avg, median, min, max, sum for a list of values."""
        if not values:
            return {"avg": 0, "median": 0, "min": 0, "max": 0, "sum": 0}
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        median = (
            sorted_vals[n // 2]
            if n % 2 == 1
            else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        )
        return {
            "avg": sum(values) / n,
            "median": median,
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }

    @staticmethod
    def print_summary():
        """Print summary statistics for all controllers."""
        print("\n" + "=" * 80)
        print("TASK METRICS SUMMARY")
        print("=" * 80)

        for controller_name, task_list in TaskMetricsCollector.statistics.items():
            if not task_list:
                continue

            print(f"\n{controller_name} ({len(task_list)} records)")
            print("-" * 70)

            task_type_data: dict[str, dict[str, list[float]]] = {}
            perf_metrics_agg: dict[str, dict[str, list[float]]] = {}

            for task_info in task_list:
                task_type = task_info["task_type"]
                if task_type not in task_type_data:
                    task_type_data[task_type] = {
                        "duration_ms": [],
                        "prompt_tokens": [],
                        "completion_tokens": [],
                        "reasoning_tokens": [],
                        "total_tokens": [],
                    }
                    perf_metrics_agg[task_type] = {}

                data = task_type_data[task_type]
                data["duration_ms"].append(task_info["duration_ms"])
                data["prompt_tokens"].append(task_info.get("prompt_tokens", 0))
                data["completion_tokens"].append(task_info.get("completion_tokens", 0))
                data["reasoning_tokens"].append(task_info.get("reasoning_tokens", 0))
                data["total_tokens"].append(task_info.get("total_tokens", 0))

                if task_info.get("perf_metrics"):
                    for key, value in task_info["perf_metrics"].items():
                        if isinstance(value, (int, float)):
                            if key not in perf_metrics_agg[task_type]:
                                perf_metrics_agg[task_type][key] = []
                            perf_metrics_agg[task_type][key].append(float(value))

            for task_type, data in task_type_data.items():
                count = len(data["duration_ms"])
                print(f"\n  {task_type} (count: {count})")

                duration_stats = TaskMetricsCollector._compute_stats(
                    data["duration_ms"]
                )
                print(
                    f"    Duration (ms):     sum={duration_stats['sum']:.2f}, "
                    f"avg={duration_stats['avg']:.2f}, "
                    f"median={duration_stats['median']:.2f}, "
                    f"min={duration_stats['min']:.2f}, max={duration_stats['max']:.2f}"
                )

                if sum(data["total_tokens"]) > 0:
                    prompt_stats = TaskMetricsCollector._compute_stats(
                        data["prompt_tokens"]
                    )
                    completion_stats = TaskMetricsCollector._compute_stats(
                        data["completion_tokens"]
                    )
                    total_stats = TaskMetricsCollector._compute_stats(
                        data["total_tokens"]
                    )

                    print(
                        f"    Prompt tokens:     sum={prompt_stats['sum']:.0f}, "
                        f"avg={prompt_stats['avg']:.1f}, "
                        f"min={prompt_stats['min']:.0f}, max={prompt_stats['max']:.0f}"
                    )
                    print(
                        f"    Completion tokens: sum={completion_stats['sum']:.0f}, "
                        f"avg={completion_stats['avg']:.1f}, "
                        f"min={completion_stats['min']:.0f}, max={completion_stats['max']:.0f}"
                    )
                    print(
                        f"    Total tokens:      sum={total_stats['sum']:.0f}, "
                        f"avg={total_stats['avg']:.1f}, "
                        f"min={total_stats['min']:.0f}, max={total_stats['max']:.0f}"
                    )

                if perf_metrics_agg[task_type]:
                    print("\n    Perf Metrics:")
                    for metric_name, values in sorted(
                        perf_metrics_agg[task_type].items()
                    ):
                        stats = TaskMetricsCollector._compute_stats(values)
                        print(
                            f"      {metric_name}: sum={stats['sum']:.2f}, "
                            f"avg={stats['avg']:.2f}, "
                            f"min={stats['min']:.2f}, "
                            f"max={stats['max']:.2f}"
                        )

        print("\n" + "=" * 80 + "\n")

    @staticmethod
    def get_statistics(controller_name: str = None) -> dict[str, list[dict[str, Any]]]:
        """Get statistics for a specific controller or all controllers."""
        if controller_name is not None:
            return {
                controller_name: TaskMetricsCollector.statistics.get(
                    controller_name, []
                )
            }
        return TaskMetricsCollector.statistics

    @staticmethod
    def get_all_records() -> list[dict[str, Any]]:
        """Get all records across all controllers as a flat list."""
        all_records = []
        for records in TaskMetricsCollector.statistics.values():
            all_records.extend(records)
        all_records.sort(key=lambda x: x.get("timestamp", 0))
        return all_records

    @staticmethod
    def export_to_json(file_path: str, controller_name: str = None):
        """Export metrics to a JSON file."""
        if controller_name is not None:
            data = TaskMetricsCollector.statistics.get(controller_name, [])
        else:
            data = TaskMetricsCollector.statistics
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def reset(controller_name: str = None):
        """Reset statistics for a specific controller or all controllers."""
        if controller_name is not None:
            if controller_name in TaskMetricsCollector.statistics:
                TaskMetricsCollector.statistics[controller_name] = []
        else:
            TaskMetricsCollector.statistics.clear()

    @staticmethod
    def get_global_info() -> Any:
        return TaskMetricsCollector.statistics
