from typing import List

import torch
from stop_sequencer import StopSequencer
from transformers import AutoModelForCausalLM, AutoTokenizer

from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class HuggingFaceDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "revision": self.revision,
        }
        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, legacy=self.tokenizer_legacy)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # assume the model is decoder-only
        self.tokenizer.padding_side = 'left'
        
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            if self.prefill and "```" in self.response_prefix:
                self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        self.model = AutoModelForCausalLM.from_pretrained(name, **kwargs)

    def is_direct_completion(self) -> bool:
        return self.direct_completion or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompts = [
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.subset, self.split, self.instruction_prefix, self.response_prefix, self.tokenizer, self.direct_completion
            )
            for prompt in prompts
        ]
        
        input_tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )["input_ids"]
        
        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature
        ret = self.model.generate(
            input_tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            num_return_sequences=num_samples,
            pad_token_id=self.tokenizer.eos_token_id,
            stop_strings=self.eos,
            tokenizer=self.tokenizer,
            **kwargs,
        )
        
        # Reshape ret into a list of lists, each sublist containing num_samples elements
        ret_chunks = [ret[i:i + num_samples] for i in range(0, len(ret), num_samples)]

        all_outputs = []
        # Process each chunk in ret_chunks
        for i, ret_chunk in enumerate(ret_chunks):
            gen_strs = self.tokenizer.batch_decode(
                ret_chunk[:, input_tokens[i].size(-1):],
                skip_special_tokens=self.skip_special_tokens,
            )
            outputs = []
            for output in gen_strs:
                min_index = 10000
                for eos in self.eos:
                    if eos in output:
                        min_index = min(min_index, output.index(eos))
                outputs.append(output[:min_index].replace("\t", "    "))
            all_outputs.append(outputs)
        return all_outputs