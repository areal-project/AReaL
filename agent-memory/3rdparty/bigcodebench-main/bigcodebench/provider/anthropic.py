import os
from typing import List
from tqdm import tqdm

import anthropic

from bigcodebench.gen.util.anthropic_request import make_auto_request
from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import make_raw_chat_prompt

class AnthropicDecoder(DecoderBase):
    def __init__(self, name: str, reasoning_budget: int = 0, reasoning_beta: str = "output-128k-2025-02-19", **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))
        self.reasoning_budget = reasoning_budget
        self.reasoning_beta = reasoning_beta

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        all_outputs = []
        for prompt in tqdm(prompts):
            outputs = []
            
            for _ in range(num_samples):
                ret = make_auto_request(
                    client=self.client,
                    model=self.name,
                    messages=[
                        {
                            "role": "user",
                            "content": make_raw_chat_prompt(
                                task_prompt=prompt,
                                subset=self.subset,
                                split=self.split,
                                instruction_prefix=self.instruction_prefix,
                                response_prefix=self.response_prefix,
                                tokenizer=None,
                            )
                        }
                    ],
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stop_sequences=self.eos,
                    reasoning_budget=self.reasoning_budget,
                    reasoning_beta=self.reasoning_beta,
                )
                if isinstance(ret, anthropic.Stream):
                    output = ""
                    for chunk in ret:
                        if chunk.type == "content_block_delta":
                            # if chunk.delta.type == "thinking_delta":
                            #     output += chunk.delta.thinking
                            if chunk.delta.type == "text_delta":
                                output += chunk.delta.text
                    outputs.append(output)
                else:
                    outputs.append(ret.content[0].text)
            all_outputs.append(outputs)
        return all_outputs

    def is_direct_completion(self) -> bool:
        return False