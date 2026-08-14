from typing import Optional
from pydantic import BaseModel
from enum import StrEnum


class CommandName(StrEnum):
    BASH = "bash"
    PYTHON = "python"
    CPP = "cpp"
    C = "c"


class CommandItem(BaseModel):
    command_name: CommandName
    script: str


class CommandExecutionResult(BaseModel):
    exit_code: Optional[int]
    output: Optional[str]
    timeout_flag: bool
