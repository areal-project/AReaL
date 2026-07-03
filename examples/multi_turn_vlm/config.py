from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class MultiTurnVLMGRPOConfig(GRPOConfig):
    max_turns: int = field(
        default=2,
        metadata={"help": "Maximum tool-calling turns per episode."},
    )
    turn_discount: float = field(
        default=1.0,
        metadata={
            "help": "Per-turn reward discount in (0, 1]. 1.0 = flat terminal reward; "
            "< 1.0 adds an early-success incentive."
        },
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "'concat' (one trajectory/episode; non-thinking VLMs, default) "
            "or 'individual' (one sample/turn; required for thinking models whose "
            "chat template strips prior <think>, e.g. Qwen3.5/3.6).",
            "choices": ["concat", "individual"],
        },
    )
    tool_format: str = field(
        default="hermes",
        metadata={
            "help": "Tool-call format the prompt advertises: 'hermes' (Qwen3-VL JSON) "
            "or 'qwen3_coder' (Qwen3.5/3.6 XML). The env parser accepts both.",
            "choices": ["hermes", "qwen3_coder"],
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # 'individual' emits a variable number of rows per episode, which breaks
        # the positional group windows used by group-level normalization.
        if self.export_style == "individual":
            for name in ("reward_norm", "adv_norm"):
                norm = getattr(self.actor, name, None)
                if norm is not None and "group" in (norm.mean_level, norm.std_level):
                    raise ValueError(
                        f"export_style='individual' is incompatible with group-level "
                        f"actor.{name} (variable rows per episode misalign positional "
                        f"group windows); set mean_level/std_level to 'batch' or null."
                    )
