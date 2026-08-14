import os
from typing import List
from tqdm import tqdm

from huggingface_hub import InferenceClient

from bigcodebench.provider.base import DecoderBase
from bigcodebench.gen.util.hf_inference_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt


class HuggingFaceInferenceDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.client = InferenceClient(
            provider="hf-inference", api_key=os.getenv("HF_INFERENCE_API_KEY")
        )

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"

        all_outputs = []

        for prompt in tqdm(prompts):
            outputs = []
            message = (
                prompt
                if self.is_direct_completion()
                else make_raw_chat_prompt(
                    task_prompt=prompt,
                    subset=self.subset,
                    split=self.split,
                    instruction_prefix=self.instruction_prefix,
                    response_prefix=self.response_prefix,
                    tokenizer=None,
                )
            )
            ret = make_auto_request(
                self.client,
                message=message,
                model=self.name,
                n=num_samples,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            outputs.append(ret)
            all_outputs.append(outputs)
        return all_outputs

    def is_direct_completion(self) -> bool:
        return self.direct_completion
