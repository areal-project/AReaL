from pydantic import BaseModel
from typing import Optional


class StartServerRequest(BaseModel):
    config_path: str


class StartServerResponse(BaseModel):
    success_flag: bool
    message: Optional[str]


class ShutdownServerRequest(BaseModel):
    pass


class ShutdownServerResponse(BaseModel):
    success_flag: bool
    message: Optional[str]
