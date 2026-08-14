import os
from typing import List
from tqdm import tqdm

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from bigcodebench.provider.base import DecoderBase
from bigcodebench.gen.util.mistral_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt

class MistralChatDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(name, **kwargs)
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

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
                        ChatMessage(
                            role="user",
                            content=make_raw_chat_prompt(
                                task_prompt=prompt,
                                subset=self.subset,
                                split=self.split,
                                instruction_prefix=self.instruction_prefix,
                                response_prefix=self.response_prefix,
                                tokenizer=None,
                                direct_completion=None,
                            )
                        )
                    ],
                    max_tokens=self.max_new_tokens,
                )
                outputs.append(ret.choices[0].message.content)
            all_outputs.append(outputs)
        return all_outputs

    def is_direct_completion(self) -> bool:
        return False