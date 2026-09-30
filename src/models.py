from pydantic import BaseModel, Field, field_validator


Familly: dict[str, str] = {
    "number": "number", "integer": "number", "int": "number",
    "float": "number", "double": "number",
    "string": "string", "str": "string",
    "boolean": "boolean", "bool": "boolean",
}


class ParamDef(BaseModel):
    """Define a function parameter's type and optional description"""
    # Accepts synonyms (integer, int, float, bool...) but always
    # normalizes to one of the 3 families the grammar can generate
    type: str
    description: str | None = None

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        """Normalize the declared type to a supported family.

        Raises a clear ValueError for truly unknown types so the
        loader can report it gracefully instead of crashing.
        """
        family = Familly.get(value.strip().lower())
        if family is None:
            raise ValueError(
                f"Unsupported parameter type: {value!r}. "
                f"Supported: {sorted(set(Familly))}"
            )
        return family


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
    # It can not have parameters and defaults a dict empty to not crash
    parameters: dict[str, ParamDef] = Field(default_factory=dict)
    returns: ReturnDef | None = None    # It can also have not a return
