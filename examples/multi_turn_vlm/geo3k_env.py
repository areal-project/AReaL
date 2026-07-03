"""Geometry3K calc_score tool-calling environment.

The model solves a geometry problem and submits its answer with a `calc_score`
tool (text-based: it emits the tool call in its output). calc_score is the SOLE
answer path: the env grades the submitted answer against the
ground truth, returns text feedback, and the model retries until correct or
`max_turns`.

Different model families emit tool calls in different formats, so the parser
accepts both and ``tool_format`` selects which the system prompt advertises:
- "hermes"      (Qwen3-VL): ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
- "qwen3_coder" (Qwen3.5/3.6): ``<function=calc_score><parameter=answer>...</parameter></function>``
"""

import json
import re

from mathruler.grader import extract_boxed_content, grade_answer

from areal.workflow.vision_env import (
    EnvResetResult,
    EnvStepResult,
    MultiTurnVisionEnv,
)

SUPPORTED_TOOL_NAMES = {"calc_score"}

# Final-turn \boxed fallback gets partial credit so calc_score stays the preferred
# path (full credit would teach the model to skip the tool and just box last turn).
FINAL_FALLBACK_CREDIT = 0.5

# qwen3_coder (Qwen3.5/3.6): <function=name>...<parameter=k>v</parameter>...</function>
XML_FUNC_RE = re.compile(r"<function=([\w.\-]+)\s*>(.*?)</function>", re.DOTALL)
XML_PARAM_RE = re.compile(
    r"<parameter=([\w.\-]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL
)


def _parse_hermes_tool_call(text: str) -> tuple[str, str] | None:
    """Hermes (Qwen3-VL): ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``.

    Each block is parsed separately so an unclosed earlier block cannot swallow
    a later valid one; the last parseable call wins.
    """
    result = None
    for chunk in text.split("<tool_call>")[1:]:
        try:
            payload = json.loads(chunk.split("</tool_call>")[0].strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        func = payload.get("function")
        func = func if isinstance(func, dict) else {}
        name = payload.get("name") or func.get("name")
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = func.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if isinstance(arguments, dict) and name:
            answer = arguments.get("answer", "")
            result = (name, "" if answer is None else str(answer).strip())
    return result


def _parse_xml_tool_call(text: str) -> tuple[str, str] | None:
    """qwen3_coder XML form; the last call wins."""
    funcs = XML_FUNC_RE.findall(text)
    if not funcs:
        return None
    name, body = funcs[-1]
    params = dict(XML_PARAM_RE.findall(body))
    answer = params.get("answer", "")
    return (name.strip(), str(answer).strip())


_TASK_INSTRUCTION = (
    "You are an expert at solving geometry problems from a figure. First reason step "
    "by step inside <think> </think> tags, keeping your reasoning concise, then commit "
    "to your best answer. Submit your answer by calling the calc_score tool, which is "
    "the ONLY way to answer. Pass ONLY the final value to the answer argument: a bare "
    "number or expression like '60' or '3/4' (write '4', not 'y = 4'), with no variable "
    "name, units, equals sign, or working. Do not give a final answer in any other "
    "form. The tool replies whether your answer matches the reference; if it is wrong, "
    "reason in a different way and call calc_score again."
)

_CALC_SCORE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calc_score",
        "description": "Submit your final answer for grading against the reference; "
        "returns whether it is correct.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "the final value only: a bare number or expression "
                    "(e.g. '60', '3/4'), with no variable name, units, or working",
                }
            },
            "required": ["answer"],
        },
    },
}

# Hermes (Qwen3-VL): a short format hint is enough — Instruct models tool-call readily.
_HERMES_SYSTEM = (
    _TASK_INSTRUCTION + "\nCall the tool in this format:\n"
    '<tool_call>{"name": "calc_score", "arguments": {"answer": "<your answer>"}}'
    "</tool_call>"
)

# qwen3_coder (Qwen3.5/3.6): a thinking model only enters tool-calling mode when it
# sees its trained tool preamble, so reproduce the exact block `tools=` would inject
# (the `tools=` API itself is an engine change we avoid). Model-specific by design.
_QWEN3_CODER_SYSTEM = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
    + json.dumps(_CALC_SCORE_SCHEMA)
    + "\n</tools>\n\nIf you choose to call a function ONLY reply in the following "
    "format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n"
    "<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>"
    "\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an "
    "inner <function=...></function> block must be nested within "
    "<tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n"
    "- You may provide optional reasoning BEFORE the function call, but NOT after\n"
    "</IMPORTANT>\n\n" + _TASK_INSTRUCTION
)

_SYSTEM_PROMPTS = {"hermes": _HERMES_SYSTEM, "qwen3_coder": _QWEN3_CODER_SYSTEM}


def _system_prompt(tool_format: str) -> str:
    return _SYSTEM_PROMPTS[tool_format]


def _strip_dataset_instruction(text: str) -> str:
    """Drop geometry3k's appended "You FIRST think ... \\boxed{}" instruction so
    the model cannot bypass calc_score with a direct boxed answer."""
    idx = text.find("You FIRST think about the reasoning process")
    return text[:idx].rstrip() if idx != -1 else text.rstrip()


class Geo3kCalcScoreEnv(MultiTurnVisionEnv):
    def __init__(self, max_turns: int = 2, tool_format: str = "hermes"):
        if tool_format not in _SYSTEM_PROMPTS:
            raise ValueError(f"tool_format must be one of {sorted(_SYSTEM_PROMPTS)}")
        self.max_turns = max_turns
        self.tool_format = tool_format
        self.ground_truth: str | None = None
        self.turns = 0
        self.correct = False

    def reset(self, data) -> EnvResetResult:
        answer = data.get("answer")
        if answer is None or str(answer).strip() in ("", "None"):
            raise ValueError(
                f"geo3k env requires a valid ground-truth 'answer', got {answer!r}"
            )
        self.ground_truth = str(answer).strip()
        self.turns = 0
        self.correct = False

        # Strip the dataset's "<think>/\boxed{}" instruction so the model can only
        # answer via calc_score, then prepend the tool system prompt.
        user_turn = data["messages_chat"][0]
        content = [
            {"type": "text", "text": _strip_dataset_instruction(c.get("text", ""))}
            if c.get("type") == "text"
            else c
            for c in user_turn["content"]
        ]
        messages_chat = [
            {
                "role": "system",
                "content": [{"type": "text", "text": _system_prompt(self.tool_format)}],
            },
            {"role": "user", "content": content},
        ]
        return EnvResetResult(messages_chat=messages_chat, images=data["images"])

    def step(self, assistant_text) -> EnvStepResult:
        self.turns += 1
        is_final_turn = self.turns >= self.max_turns

        tool_call = self._extract_tool_call(assistant_text)

        # (1) No tool call: calc_score is the only path on non-final turns; on the
        # final turn fall back to a \boxed answer so a correct solution isn't lost.
        if tool_call is None:
            if is_final_turn:
                return self._final_fallback(assistant_text)
            return EnvStepResult(
                observation=self._no_tool_feedback(), reward=0.0, done=False
            )

        name, parsed_answer = tool_call

        # (2) Unsupported tool name.
        if name not in SUPPORTED_TOOL_NAMES:
            if is_final_turn:
                return self._final_fallback(assistant_text)
            return EnvStepResult(
                observation=UNSUPPORTED_TOOL_FEEDBACK, reward=0.0, done=False
            )

        # (3) Missing answer argument.
        if parsed_answer == "":
            if is_final_turn:
                return self._final_fallback(assistant_text)
            return EnvStepResult(
                observation=ANSWER_MISSING_FEEDBACK, reward=0.0, done=False
            )

        # (4) Grade the submitted tool answer (the model's chosen submission).
        score = self._score_answer(parsed_answer)
        self.correct = score == 1.0
        done = self.correct or is_final_turn
        if done:
            return EnvStepResult(observation=None, reward=score, done=True)
        return EnvStepResult(
            observation=self._wrong_feedback(score, parsed_answer),
            reward=0.0,
            done=False,
        )

    def get_metrics(self):
        return {"turns": float(self.turns), "acc": 1.0 if self.correct else 0.0}

    # ----- helpers -----

    def _extract_tool_call(self, text):
        # The advertised format wins over the other (a draft in another format
        # discussed mid-reasoning must not outrank the real call).
        parsers = (_parse_hermes_tool_call, _parse_xml_tool_call)
        if self.tool_format == "qwen3_coder":
            parsers = (_parse_xml_tool_call, _parse_hermes_tool_call)
        for parse in parsers:
            found = parse(text)
            if found is not None:
                return found
        return None

    def _final_fallback(self, assistant_text) -> EnvStepResult:
        """Last-turn safety net: if no usable calc_score call was produced on the
        final turn, still credit a \\boxed{} answer so reward tracks correctness,
        not tool-format compliance. Partial credit keeps calc_score preferred."""
        boxed = extract_boxed_content(assistant_text)
        raw = self._score_answer(boxed) if boxed and boxed != "None" else 0.0
        self.correct = raw == 1.0
        return EnvStepResult(
            observation=None, reward=raw * FINAL_FALLBACK_CREDIT, done=True
        )

    def _no_tool_feedback(self) -> str:
        # Entering the final turn, reveal the \boxed fallback so a correct answer
        # isn't lost to a tool-format slip on the last attempt.
        if self.turns >= self.max_turns - 1:
            return (
                "You did not call the calc_score tool. This is your LAST attempt: "
                "call calc_score with your answer, or give your final answer as "
                "\\boxed{your answer}."
            )
        return NO_TOOL_FEEDBACK

    def _score_answer(self, answer) -> float:
        answer = (answer or "").strip()
        if not answer or answer == "None":
            return 0.0
        if "\\boxed" in answer:
            inner = extract_boxed_content(answer)
            if inner and inner != "None":
                answer = inner
        try:
            return 1.0 if grade_answer(answer, self.ground_truth) else 0.0
        except Exception:
            return 0.0

    def _wrong_feedback(self, score, parsed_answer) -> str:
        # Cap the echoed answer so a degenerate submission cannot blow the
        # workflow's per-turn observation budget.
        if len(parsed_answer) > 80:
            parsed_answer = parsed_answer[:77] + "..."
        # One turn before the hard limit, tell the model it is the last attempt.
        last_chance = self.max_turns >= 2 and self.turns >= self.max_turns - 1
        base = (
            f"calc_score result: {score}. Parsed answer '{parsed_answer}' does not "
            "match the reference. Your answer is wrong."
        )
        if last_chance:
            return (
                base + " This is your last attempt: reason in a different way and "
                "call calc_score once more, or give your final answer as "
                "\\boxed{your answer}."
            )
        return base + " Reason in a different way and call calc_score again."


# Edge-case feedback. Plain prose only (no literal tool-call tag) so per-turn
# observations carry no special tokens.
NO_TOOL_FEEDBACK = (
    "You did not call the calc_score tool. Submit your answer by calling calc_score "
    "with your candidate answer; that is the only way to answer."
)
UNSUPPORTED_TOOL_FEEDBACK = (
    "That tool is not available. Check your answer by calling the calc_score "
    "tool with an 'answer' argument."
)
ANSWER_MISSING_FEEDBACK = (
    "A tool call was detected but no 'answer' argument was provided. Call the "
    "calc_score tool with your candidate answer."
)
