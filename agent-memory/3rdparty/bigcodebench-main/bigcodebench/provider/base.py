from abc import ABC, abstractmethod
from typing import List

from bigcodebench.provider.utility import EOS


class DecoderBase(ABC):
    def __init__(
        self,
        name: str,
        subset: str,
        split: str,
        temperature: float = 0.8,
        max_new_tokens: int = 1280,
        revision: str = "main",
        dtype: str = "bfloat16",  # default
        direct_completion: bool = False,
        trust_remote_code: bool = False,
        tokenizer_name: str = None,
        tokenizer_legacy: bool = False,
        instruction_prefix: str = None,
        response_prefix: str = None,
        prefill: bool = True,
    ) -> None:
        print("Initializing a decoder model: {} ...".format(name))
        self.name = name
        self.subset = subset
        self.split = split
        self.temperature = temperature
        self.eos = EOS
        self.skip_special_tokens = False
        self.max_new_tokens = max_new_tokens
        self.dtype = dtype
        self.revision = revision
        self.direct_completion = direct_completion
        self.trust_remote_code = trust_remote_code
        self.tokenizer_name = tokenizer_name
        self.tokenizer_legacy = tokenizer_legacy
        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix
        self.prefill = prefill

    @abstractmethod
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        pass

    @abstractmethod
    def is_direct_completion(self) -> bool:
        pass

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name