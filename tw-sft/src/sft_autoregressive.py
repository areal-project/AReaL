import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
import transformers
import trl
from datasets import load_dataset
from trl import SFTTrainer
from utils import add_and_init_special_tokens, SequentialSFTTrainer


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen3-8B-131072")
    template_name: Optional[str] = field(default=None)
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="ThreadWeaver")
    train_file_path: Optional[str] = field(
        default="data/training_dataset_autoregressive"
    )
    no_special_tokens: bool = field(default=False)
    dagger: bool = field(default=False)
    attn_implementation: Optional[str] = field(default=None)
    no_shuffle: bool = field(default=True)

    def __post_init__(self):
        os.environ["WANDB_PROJECT"] = self.wandb_project


def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    args.max_length = config.block_size
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    kwargs = {}
    if config.attn_implementation is not None:
        # https://huggingface.co/docs/transformers/en/attention_interface
        # Examples are sdpa, flash_attention_2, flex_attention
        kwargs["attn_implementation"] = config.attn_implementation
    if "70B" in config.model_name:
        # Removed "low_cpu_mem_usage": True, for 70B, since by default we are in FSDP,
        # it's more efficient to do  "cpu_ram_efficient_loading": true, in fsdp_config.json
        kwargs = {
            "device_map": "auto",
            "torch_dtype": "auto",
            "attn_implementation": "flash_attention_2",
            "use_cache": False,
        }
        model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model_name, **kwargs
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)

    dataset = load_dataset(config.train_file_path)

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name, use_fast=True
    )
    template_name = config.template_name
    if template_name is None:
        raise ValueError(
            "Please specify a template type using --template-name. Options are 'llama', 'ds', or 'qwen'."
        )
        # if "Llama" in config.model_name:
        #     template_name = "llama"
        # elif "DeepSeek" in config.model_name:
        #     template_name = "ds"
        # elif "Qwen" in config.model_name:
        #     template_name = "qwen"
        # else:
        #     raise ValueError(
        #         f"Unsupported model {config.model_name}. Please use a Llama or Qwen or DS model."
        #     )
    if template_name == "llama":
        print(f"Using Llama templates for {config.model_name}")
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif template_name == "ds":
        print(f"Using DeepSeek templates for {config.model_name}")
        instruction_template = "<｜User｜>"
        response_template = "<｜Assistant｜>"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    elif template_name == "qwen":
        print(f"Using Qwen templates for {config.model_name}")
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"
    else:
        raise ValueError(
            f"Unsupported model {config.model_name}. Please use a Llama or Qwen or DS model."
        )

    if not config.no_special_tokens:
        print(f"Adding special tokens for {config.model_name}")
        add_and_init_special_tokens(model, tokenizer)
    else:
        print(f"Not adding special tokens for {config.model_name}")

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    if args.dataset_text_field is None:
        args.dataset_text_field = "text"
    args.max_seq_length = 131072

    if "TENSORBOARD_DIR" in os.environ and "wandb" in args.report_to:
        # args.report_to is a list, so we need to append to it
        args.report_to.append("tensorboard")
        args.logging_dir = os.environ["TENSORBOARD_DIR"]

    print(f"args.dataset_text_field: {args.dataset_text_field}")
    print(f"args.report_to: {args.report_to}")
    print(f"args.logging_dir: {args.logging_dir}")

    if config.no_shuffle:
        print("Shuffling is DISABLED. Please shuffle and concatenate the dataset on your own.")
        trainer = SequentialSFTTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
            args=args,
            processing_class=tokenizer,
            data_collator=collator,
        )
    else:
        print("Shuffling is ENABLED.")
        trainer = SFTTrainer(
            model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"] if "test" in dataset else dataset["train"],
            args=args,
            processing_class=tokenizer,
            data_collator=collator,
        )

    trainer.train()

    trainer.save_model(output_dir=args.output_dir)
    if trainer.is_local_process_zero():
        tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
