import os
from typing import List
from tqdm import tqdm

from google import genai

from bigcodebench.provider.base import DecoderBase
from bigcodebench.gen.util.google_request import make_auto_request
from bigcodebench.provider.utility import make_raw_chat_prompt


class GoogleDecoder(DecoderBase):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self.model = name
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if do_sample:
            assert self.temperature > 0, "Temperature must be positive for sampling"
        
        all_outputs = []
        
        for prompt in tqdm(prompts):
            outputs = []
            message = make_raw_chat_prompt(
                task_prompt=prompt,
                subset=self.subset,
                split=self.split,
                instruction_prefix=self.instruction_prefix,
                response_prefix=self.response_prefix,
                tokenizer=None,
            )
            ret = make_auto_request(
                model=self.model,
                client=self.client,
                message=message,
                n=num_samples,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            for candidate in ret.candidates:
                parts = candidate.content.parts
                if parts:
                    outputs.append(parts[0].text)
                else:
                    print("Empty response!")
                    outputs.append("")
                    print(f"{candidate.safety_ratings = }")
            all_outputs.append(outputs)
        return all_outputs

    def is_direct_completion(self) -> bool:
        return False