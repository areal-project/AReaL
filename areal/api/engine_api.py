# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import PerfTracerConfig
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    LocalInfServerInfo,
    ModelRequest,
    ModelResponse,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)

if TYPE_CHECKING:
    from areal.api.workflow_api import WorkflowLike
    from areal.infra import WorkflowExecutor
    from areal.utils.data import MicroBatchList


class TrainEngine(abc.ABC):
    @abc.abstractmethod
    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        """Initialize PyTorch distributed communication groups.

        Parameters
        ----------
        parallel_strategy : ParallelStrategy, optional
            The parallel strategy configuration for distributed training, by default None
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def initialize(self, *args, **kwargs):
        """Initialize environments for distributed training and load models.

        This method should be called after `create_process_group`.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_group(self) -> dist.ProcessGroup:
        """Get the data parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The data parallel communication group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_rank(self) -> int:
        """Get the rank of the current process in the data parallel group.

        Returns
        -------
        int
            The rank of the current process in the data parallel group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def data_parallel_world_size(self) -> int:
        """Get the world size of the data parallel group.

        Returns
        -------
        int
            The world size of the data parallel group
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def current_data_parallel_head(self) -> int:
        """Get the current data parallel head rank.

        Returns
        -------
        int
            The rank of the current data parallel head
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_data_parallel_head(self) -> bool:
        """Check if the current rank is the data parallel head of the current engine.

        Returns
        -------
        bool
            True if the current rank is the data parallel head, False otherwise
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        """Get the context and model parallel communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The context and model parallel communication group
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def cpu_group(self) -> dist.ProcessGroup:
        """Get the CPU communication group of this engine.

        Returns
        -------
        dist.ProcessGroup
            The CPU communication group
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory of models."""

    @property
    @abc.abstractmethod
    def initialized(self) -> bool:
        """Check if the engine has been initialized.

        Returns
        -------
        bool
            True if initialize() has been called successfully, False otherwise
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self, mode: bool = True):
        """Set the engine to training mode.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the engine to training mode, by default True
        """
        raise NotImplementedError()

    def eval(self):
        """Set the engine to evaluation mode.

        This is a convenience method that calls `self.train(False)`.
        """
        return self.train(False)

    @abc.abstractmethod
    def update_weights(self, meta: WeightUpdateMeta):
        """Update weights to the inference engine in a blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        """Connect to an inference engine for online training.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine to connect to
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        """Submit a batch of requests and wait for results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.
        Should note that this is a simple rollout engine method forwarding with
        distributed data management.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries.
        workflow : WorkflowLike
            The workflow to use for rollout generation.
        workflow_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the workflow constructor, by default None.
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory contains tensors with shape [group_size, seqlen, ...].
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepare a batch of data for training from a dataloader.

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The dataloader to fetch data from.
        workflow : WorkflowLike
            The workflow to use for rollout generation.
        workflow_kwargs : dict[str, Any] | None, optional
            Keyword arguments to pass to the workflow constructor, by default None.
        should_accept_fn : Callable[[dict[str, Any]], bool] | str | None, optional
            A function to filter trajectories, by default None.
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. The method will stop collecting
            when (accepted + rejected) >= batch_size, returning only accepted results.
            This results in variable-sized batches of valid data. Default is False.

        Returns
        -------
        dict[str, Any]
            The prepared batch data.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_version(self, version: int):
        """Set the current weight version in the training engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_version(self) -> int:
        """Get the current weight version in the training engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, meta: SaveLoadMeta):
        """Save model weights and optimizer states for later use.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to save
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, meta: SaveLoadMeta):
        """Load model weights and optimizer states from a file.

        Parameters
        ----------
        meta : SaveLoadMeta
            Metadata containing information about where and how to load
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_zero_grad(self):
        """Zero out all gradients in the optimizer."""
        raise NotImplementedError()

    @abc.abstractmethod
    def optimizer_step(self):
        """Perform a single optimization step.

        Returns
        -------
        dict[str, float]
            Training statistics containing ``update_successful``, ``grad_norm``, and ``lr``.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def lr_scheduler_step(self):
        """Advance the learning rate scheduler by one step."""
        raise NotImplementedError()

    def step_lr_scheduler(self):
        """This is an alias for `lr_scheduler_step()`."""
        return self.lr_scheduler_step()

    @abc.abstractmethod
    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        """Process micro-batches through forward and optionally backward pass.

        Parameters
        ----------
        mb_list : MicroBatchList
            The micro-batch list, which is iterable and yields MicroBatchItem tuples.
        process_output_fn : Callable[[torch.Tensor, dict[str, Any]], torch.Tensor | None]
            A function that processes the model output (logits) and returns the loss tensor.
            If the returned loss is not None, backward() will be called on it.
            Results can be collected via closure if needed.
            Signature: ``(logits: Tensor, inputs: dict) -> loss | None``
        forward_only : bool, optional
            If True, skip backward pass. Default is False.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        """Update the model with a batch of data and a loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : list[dict[str, Any]] | dict[str, Any]
            Input data for model forward pass and loss computation.
            Preferred format is ``list[dict[str, Any]]`` (trajectory list).
            Backward compatibility: a pre-batched ``dict[str, Any]`` is
            also accepted.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        dict[str, float]
            Scalar statistics after training, e.g., the current learning rate,
            gradient norm, etc.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def eval_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        """Evaluate the model using the forward pass and loss function.

        Note
        ----
        The loss_fn should process packed 1D inputs, instead of 2D inputs.

        Parameters
        ----------
        input_ : list[dict[str, Any]] | dict[str, Any]
            Input data for model forward pass and loss computation.
            Preferred format is ``list[dict[str, Any]]`` (trajectory list).
            Backward compatibility: a pre-batched ``dict[str, Any]`` is
            also accepted.
        loss_fn : Callable[..., torch.Tensor]
            The loss function. For actor (is_critic=False), it receives
            (logprobs, entropy, input_data). For critic (is_critic=True),
            it receives (values, input_data). Returns a scalar normalized loss.
        loss_weight_fn : Callable[[dict[str, Any]], torch.Tensor]
            A function used to calculate the weight of each micro-batch. Since
            loss_fn normalizes the loss for a micro-batch, we need a corresponding
            weight for each micro-batch to normalize the loss globally. The weight
            is usually the number of response tokens in the batch.

        Returns
        -------
        torch.Tensor or None
            A scalar loss or None. The evaluation statistics should be aggregated
            with `stats_tracker`.
        """
        raise NotImplementedError()

    @torch.no_grad()
    @abc.abstractmethod
    def forward_batch(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run the forward pass or inference on the model.

        Note
        ----
        This operation is gradient-free.

        Parameters
        ----------
        input_ : list[dict[str, Any]] | dict[str, Any]
            Input data for model forward pass. Redundant entries are allowed.
            ``list[dict[str, Any]]`` and pre-batched ``dict[str, Any]``
            are both supported.
        output_seqlens : list[int], optional
            The desired output sequence lengths. If None, assumes that the output
            has the same lengths as inputs, by default None.
        aggregate_fn : Callable[[list[torch.Tensor]], torch.Tensor], optional
            A function to aggregate micro-batched outputs, by default torch.cat.
            It should preserve batch dimension 0.

        Returns
        -------
        torch.Tensor | list[torch.Tensor]
            Batched tensor output for dict input.
            Per-trajectory tensor list for list input.
            For actor (is_critic=False), return logprobs tensors.
            For critic (is_critic=True), return value tensors.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward(
        self,
        input_: list[dict[str, Any]] | dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[torch.Tensor]], torch.Tensor] = torch.cat,
    ) -> torch.Tensor | list[torch.Tensor]:
        return self.forward_batch(input_, output_seqlens, aggregate_fn)

    @abc.abstractmethod
    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded in this engine process.

        Note
        ----
        Statistics will be all-reduced across the data parallel group
        and broadcasted from the last pipeline parallel stage.

        Returns
        -------
        dict[str, float]
            The exported scalar statistics.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def onload(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def offload(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_device_stats(self) -> DeviceRuntimeInfo:
        raise NotImplementedError()

    def start_memory_profile(self, max_entries: int = 100000) -> None:
        pass

    def stop_memory_profile(self, snapshot_dir: str) -> None:
        pass

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        """Save performance tracer data.

        Parameters
        ----------
        step : int, optional
            The current training step number, by default None
        force : bool, optional
            If True, force save regardless of internal conditions, by default False
        """

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        """Configure performance tracer.

        Parameters
        ----------
        config : PerfTracerConfig
            Configuration for the performance tracer.
        rank : int
            Rank of the current process within its role.
        role : str
            Role of this process. "master" by default or "actor",
            "ref", "rollout", etc. in RPC workers.
        """


class InferenceEngine(abc.ABC):
    def initialize(self, *args, **kwargs):
        """Initialize environments and launch the background thread for asynchronous distributed inference.

        For remote inference engines, this serves as a client and connects to the inference servers.
        For local inference engines, this creates an LLM engine on the local GPU.

        Parameters
        ----------
        *args
            Variable length argument list
        **kwargs
            Arbitrary keyword arguments
        """
        raise NotImplementedError()

    def destroy(self):
        """Destroy the engine and release GPU memory for the local inference engine."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def initialized(self) -> bool:
        """Check if the engine has been initialized.

        Returns
        -------
        bool
            True if initialize() has been called successfully, False otherwise
        """
        raise NotImplementedError()

    @property
    def workflow_executor(self) -> WorkflowExecutor:
        """Get the workflow executor of the inference engine."""
        raise NotImplementedError()

    def launch_server(self, server_args: dict[str, Any]) -> LocalInfServerInfo:
        """Launch a local inference server via subprocess and return its connection info.

        By default, an `InferenceEngine` instance acts as a client that connects to an existing
        remote inference server without occupying GPU resources. This is the typical usage in
        SPMD mode, where each training process has an attached inference client.

        This method enables launching a local inference server process, which is useful for:

        1. **Single-controller mode**: Launch a local server to serve the `InferenceEngine`
           instance with direct GPU worker control.

        2. **Standalone inference**: Use AReaL's inference engine in independent scripts or notebooks
           for running agentic workflows without managing separate server processes.

        Parameters
        ----------
        server_args : dict[str, Any]
            CLI arguments for the inference server (e.g., model path, GPU indices,
            port numbers, backend-specific settings)

        Returns
        -------
        LocalInfServerInfo
            Information about the launched server, including connection details and process metadata

        See Also
        --------
        teardown_server : Teardown the server launched by this method
        """
        raise NotImplementedError()

    def teardown_server(self):
        """Teardown the inference server launched by `launch_server`."""
        raise NotImplementedError()

    async def agenerate(self, req: ModelRequest) -> ModelResponse:
        """Asynchronously generate a response for the given request.

        Parameters
        ----------
        req : ModelRequest
            The model request containing input data and generation parameters

        Returns
        -------
        ModelResponse
            The generated response from the model
        """
        raise NotImplementedError()

    def init_weights_update_group(
        self, meta: WeightUpdateMeta, rank_ids: list[int] | None = None
    ) -> Future[None]:
        """Initialize the weight update process group for distributed weight updates.

        This method should be called before performing any weight updates to ensure
        that the necessary communication groups are set up correctly.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update, such as the
            type of communication backend and allocation mode.

        rank_ids : list[int] | None, optional
            Rank_ids per server/worker in the weight-update group. If None, the
            implementation should default to using the server index order
            (e.g. enumerate(addresses)).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.

        Returns
        -------
        Future[None]
            A future object representing the asynchronous initialization operation.
        """
        raise NotImplementedError()

    def update_weights_from_distributed(
        self, meta: WeightUpdateMeta, param_specs: list[ParamSpec]
    ) -> Future[None]:
        """Update weights in the inference engine in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update
        param_specs : List[ParamSpec]
            A list of parameter specifications for the weights to be updated

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def update_weights_from_disk(self, meta: WeightUpdateMeta) -> Future[None]:
        """Update weights in the inference engine from disk in a non-blocking manner.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Metadata containing information about the weight update

        Returns
        -------
        Future[None]
            A future object representing the asynchronous weight update operation
        """
        raise NotImplementedError()

    def set_version(self, version: int) -> None:
        """Set the current weight version in the inference engine.

        Parameters
        ----------
        version : int
            The weight version number to set
        """
        raise NotImplementedError()

    def get_version(self) -> int:
        """Get the current weight version in the inference engine.

        Returns
        -------
        int
            The current weight version number
        """
        raise NotImplementedError()

    def compute_logp(self, data: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute token log-probabilities for teacher distillation.

        Implementations support this as an inference-side scoring API.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_logp()."
        )

    def submit(
        self,
        data: dict[str, Any],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable | None = None,
        group_size: int = 1,
        task_id: int | None = None,
        is_eval: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> int:
        """Submit a request to the inference engine and return immediately.

        Should be used together with subsequent `wait`.

        Parameters
        ----------
        data : dict[str, Any]
            The input data for rollout. Used by the user's customized workflow implementation.
        workflow : WorkflowLike
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
            - An agent workflow (any class with async run() method)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function used to decide whether to accept a specific trajectory, i.e., dynamic filtering.
            It takes a complete trajectory output by the workflow, and returns a bool, by default None.
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        task_id : int, optional
            The task ID to use. If None, a new task ID will be generated internally.
        is_eval : bool, optional
            Whether this is an evaluation workflow. Affects variables like trajectory dump path
            and statistics keys. By default False.

        Returns
        -------
        int
            The id assigned to this task
        """
        raise NotImplementedError()

    def wait(
        self, count: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> list[dict[str, Any] | None]:
        """Wait for a specified number of requests to complete, with a timeout.

        Should be used together with preceding `submit`.

        Parameters
        ----------
        count : int
            The number of accepted trajectories to wait for
        timeout : float, optional
            Timeout in seconds. Exceeding the timeout will raise a `TimeoutError`, by default None
        raise_timeout : bool, optional
            Whether to raise a `TimeoutError` when the timeout is exceeded,
            otherwise return an empty list, by default True

        Returns
        -------
        list[dict[str, Any] | None]
            A list of trajectory dictionaries. Each element may be None for rejected trajectories.

        Raises
        ------
        TimeoutError
            If the timeout is exceeded before enough trajectories are collected
        """
        raise NotImplementedError()

    def wait_for_task(
        self, task_id: int, timeout: float | None = None, raise_timeout: bool = True
    ) -> dict[str, Any] | None:
        """Wait for a specific task to complete by task_id.

        Parameters
        ----------
        task_id : int
            The task ID returned by submit()
        timeout : float | None, optional
            Timeout in seconds, by default None
        raise_timeout : bool, optional
            Whether to raise TimeoutError on timeout, by default True

        Returns
        -------
        dict[str, Any] | None
            Trajectory dict, or None if rejected or timeout with raise_timeout=False

        Raises
        ------
        ValueError
            If task_id was never submitted or already consumed
        TimeoutError
            If timeout expires and raise_timeout=True
        """
        raise NotImplementedError()

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        """Submit a batch of requests to the inference engine and wait for the results.

        This method does not support asynchronous rollout and should be used for offline
        data collection or debugging, not in production experiments.

        See `workflow_api.py` for concrete implementation.

        Parameters
        ----------
        data : list[dict[str, Any]]
            A list of input data dictionaries for rollout
        workflow : WorkflowLike
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
            - An agent workflow (any class with async run() method)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        raise NotImplementedError()

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
        reward_normalization: bool = False,
        drop_incomplete_group: bool = False,
    ) -> list[dict[str, Any]]:
        """Asynchronously submit and wait until a full batch is ready with controlled staleness.

        See `workflow_api.py` for concrete implementation.

        .. warning::

            This method caches an internal data generator on the first call.
            The ``dataloader``, ``workflow``, ``workflow_kwargs``, ``group_size``,
            and ``should_accept_fn`` parameters are captured at the first invocation
            and reused in all subsequent calls. Passing different arguments in
            later calls will **not** take effect.

            If you need to switch configurations mid-training, consider:

            - Using a separate inference engine instance
            - Using the :meth:`submit` / :meth:`wait` pattern for finer control

        Parameters
        ----------
        dataloader : StatefulDataLoader
            The data loader to pull data from for batch preparation
        workflow : WorkflowLike
            The workflow to use for rollout generation. Can be:

            - An instance of RolloutWorkflow (for sharing resources between rollouts)
            - A RolloutWorkflow class type (will be instantiated with workflow_kwargs)
            - A string module path like "areal.workflow.rlvr.RLVRWorkflow" (will be imported
              and instantiated with workflow_kwargs)
            - An agent workflow (any class with async run() method)
        workflow_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the workflow constructor when workflow is a type or string.
            Required when workflow is a type or string, ignored when workflow is an instance.
            By default None.
        should_accept_fn : Callable, optional
            A function to decide whether to accept a trajectory, by default None
        group_size : int, optional
            Number of times to run the workflow per input and concatenate results.
            Default is 1 (no grouping).
        dynamic_bs : bool, optional
            If True, enables dynamic batch sizing. The method will stop collecting
            when (accepted + rejected) >= batch_size, returning only accepted results.
            This results in variable-sized batches of valid data. Default is False.

        Returns
        -------
        list[dict[str, Any]]
            A list of trajectory dictionaries, one per accepted rollout result.
            Each trajectory is a dict of tensors with shape [batch_size, seqlen, ...],
            where batch_size can vary per trajectory depending on the workflow output.
        """
        raise NotImplementedError()

    def pause_generation(self):
        """Pause the generation of inference engine.

        Used during updating weights from distributed or disk.
        """
        raise NotImplementedError()

    def continue_generation(self):
        """Continue the generation of inference engine."""
        raise NotImplementedError()

    def pause(self):
        """Pause request submission for async rollout.

        Used during evaluation to prevent data over-generation.
        """
        raise NotImplementedError()

    def resume(self):
        """Resume request submission for async rollout."""
        raise NotImplementedError()

    def offload(self):
        """Offload model from GPU to CPU for inference engine."""
        raise NotImplementedError()

    def onload(self, tags: list[str] | None = None):
        """Onload model from CPU to GPU for inference engine.

        Parameters
        ----------
        tags : list[str], optional
            Tags to onload specific components. If None, onloads all components.
        """
        raise NotImplementedError()

    def export_stats(self) -> dict[str, float]:
        """Export the statistics recorded during workflow execution in the process.

        Workflow should only record scalar metrics like "rewards".
        These metrics will be reduced in the controller side.

        Note
        ----
        This method should be only called by the controller.

        Returns
        -------
        dict[str, float]
            The recorded scalar statistics.
        """
        raise NotImplementedError()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        """Save performance tracer data.

        Parameters
        ----------
        step : int, optional
            The current training step number, by default None
        force : bool, optional
            If True, force save regardless of internal conditions, by default False
        """

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        """Configure performance tracer.

        Parameters
        ----------
        config : PerfTracerConfig
            Configuration for the performance tracer.
        rank : int
            Rank of the current process within its role.
        role : str
            Role of this process. "master" by default or "actor",
            "ref", "rollout", etc. in RPC workers.
        """
