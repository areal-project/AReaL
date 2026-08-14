import os
from typing import List
from tqdm import tqdm
import openai

from bigcodebench.gen.util.openai_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import concurrent_call

class OpenAIChatDecoder(DecoderBase):
    def __init__(self, name: str, base_url=None, reasoning_effort="medium", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
    
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        messages = [make_raw_chat_prompt(
            task_prompt=prompt,
            subset=self.subset,
            split=self.split,
            instruction_prefix=self.instruction_prefix,
            response_prefix=self.response_prefix,
            tokenizer=None,
        ) for prompt in prompts]
        # use concurrency based batching for o1 and deepseek models
        if any(self.name.startswith(model) or self.name.endswith(model) for model in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):
            return self._codegen_batch_via_concurrency(messages, num_samples)

        return self._codegen_api_batch(messages, num_samples)

    def _codegen_api_batch(self, messages: List[str], num_samples: int) -> List[str]:
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=self.base_url
        )
        
        all_outputs = []
        for message in tqdm(messages):
            ret = make_auto_request(
                client,
                message=message,
                model=self.name,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                n=num_samples,
            )
            outputs = []
            for item in ret.choices:
                outputs.append(item.message.content)
            all_outputs.append(outputs)
        return all_outputs

    def _codegen_batch_via_concurrency(self, messages: List[str], num_samples: int) -> List[str]:
        batches = concurrent_call(
            num_samples, self._codegen_api_batch, messages, num_samples=1
        )
        return [[element for sublist in item for element in sublist] for item in zip(*batches)]

    def is_direct_completion(self) -> bool:
        return False