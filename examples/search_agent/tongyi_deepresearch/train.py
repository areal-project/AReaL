from __future__ import annotations

import hashlib
import json
import sys
import uuid
from pathlib import Path

from datasets import load_dataset
from openai import AsyncOpenAI
from transformers import PreTrainedTokenizerFast

from areal import PPOTrainer, workflow_context
from areal.api import RolloutWorkflow
from areal.api.cli_args import (
    GenerationHyperparameters,
    load_expr_config,
)
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer

try:  # Package-style relative import (works if executed via -m with package context)
    from .react_agent import MultiTurnReactAgent  # type: ignore
except ImportError:  # Fallback when executed directly (no package parent known)
    module_dir = Path(__file__).parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from react_agent import MultiTurnReactAgent  # type: ignore

from examples.search_agent.tongyi_deepresearch.config import AgentRLConfig

worker_id = uuid.uuid4().hex[:4]

logger = logging.getLogger(f"ASearcher-Reasoning @ {worker_id}")


def hash(numbers):
    """Hash an entire list of integers as a single string"""
    # Convert list to string representation
    list_str = json.dumps(numbers, sort_keys=True)  # sort_keys for consistency
    return hashlib.sha256(list_str.encode()).hexdigest()


class TongyiDeepResearchReactWorkflow(RolloutWorkflow):
    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        max_tokens: int = 32768,
        max_llm_calls_per_run: int = 100,
        judge_engine_addr: str | None = None,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

        self.judge_engine_addr = judge_engine_addr
        self.max_llm_calls_per_run = max_llm_calls_per_run

    async def arun_episode(self, engine, data):
        http_client = await workflow_context.get_httpx_client()
        async with AsyncOpenAI(
            base_url=self.judge_engine_addr, api_key="EMPTY", http_client=http_client
        ) as judge_client:
            agent = MultiTurnReactAgent(
                tokenizer=self.tokenizer,
                max_tokens_per_turn=self.gconfig.max_new_tokens,
                max_llm_calls_per_run=self.max_llm_calls_per_run,
                max_total_tokens=self.max_tokens,
                judge_client=judge_client,
            )
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = str(qid) or uuid.uuid4().hex
            data["qid"] = qid

            client = ArealOpenAI(
                engine=engine, tokenizer=self.tokenizer, chat_template_type="concat"
            )

            # Collect single trajectory
            stats = await agent.make_trajectory(
                data=data,
                client=client,
            )
            stats_tracker.get(workflow_context.stat_scope()).scalar(**stats)

            completion_with_rewards = client.export_interactions(style="concat")
            assert len(completion_with_rewards) == 1, len(completion_with_rewards)
            return completion_with_rewards


def get_search_dataset(dataset_path, tokenizer):
    dataset = load_dataset(
        path="json",
        split="train",
        data_files=dataset_path,
    )
    # dataset = dataset.filter(lambda x: len(tokenizer.encode(x["question"])) <= 1024)
    return dataset


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset
    train_dataset = get_search_dataset(config.train_dataset.path, tokenizer=tokenizer)

    # Create trainer (no valid_dataset for this example)
    with PPOTrainer(config, train_dataset, valid_dataset=None) as trainer:
        # Launch judge LLM
        from areal.api import ModelAllocation
        from areal.api.cli_args import SGLangConfig

        judge_alloc = ModelAllocation.from_str(config.judge_engine.backend)
        assert judge_alloc.backend == "sglang"
        server_args = SGLangConfig.build_args(
            sglang_config=config.sglang,
            tp_size=judge_alloc.parallel.tp_size,
            base_gpu_id=0,
        )
        config.judge_engine.max_head_offpolicyness = int(1e12)
        from areal.engine.sglang_remote import RemoteSGLangEngine

        controller = None
        try:
            controller = RemoteSGLangEngine.as_controller(
                config.judge_engine, trainer.scheduler
            )
            controller.initialize(role="judge_engine", server_args=server_args)
            controller.start_proxy()
            controller.start_proxy_gateway()

            judge_engine_addr = controller.proxy_gateway_addr

            workflow_kwargs = dict(
                gconfig=config.gconfig,
                tokenizer=config.tokenizer_path,
                max_tokens=config.max_tokens_per_trajectory,
                max_llm_calls_per_run=config.max_llm_calls_per_run,
                judge_engine_addr=judge_engine_addr,
            )

            # Run training
            trainer.train(
                workflow="examples.search_agent.tongyi_deepresearch.train.TongyiDeepResearchReactWorkflow",
                workflow_kwargs=workflow_kwargs,
                eval_workflow=None,
            )
        finally:
            if controller is not None:
                controller.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
