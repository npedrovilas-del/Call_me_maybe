from typing import Literal

from pydantic import BaseModel, Field


class ParamDef(BaseModel):
    """Define a function parameter's type and optional description"""
    type: Literal["number", "string", "boolean"] # Restrict to one of this stings the parameter type
    description: str | None = None


class ReturnDef(BaseModel):
    """Describe a function's return type"""
    type: str


class PromptItem(BaseModel):
    """Represent an input prompt"""
    prompt: str


class CallResult(BaseModel):
    """Represent a generated function call for an input prompt"""
    prompt: str
    name: str
    parameters: dict[str, object]


class FunctionDef(BaseModel):
    """Define a function's name, description, parameters, and return type"""
    name: str
    description: str
    parameters: dict[str, ParamDef] = Field(default_factory=dict) # It can not have parameters and defaults a dict empty to not crash
    returns: ReturnDef | None = None # It can also have not a return
