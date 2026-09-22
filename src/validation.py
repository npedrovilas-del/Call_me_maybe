import json
import os
import sys

from llm_sdk import Small_LLM_Model
from src.models import CallResult, FunctionDef, PromptItem
from src.generate import build_prompt, generate_call, to_ids
from src.vocab import get_token_text


def validate_result(
    obj: dict[str, object], functions: list[FunctionDef],
) -> CallResult:
    """Last verification of the output data to the JSON before is creation.
    Check parameters and functions name to make sure it matches"""
    func = next(f for f in functions if f.name == obj["name"])
    params = obj["parameters"]
    if not isinstance(params, dict):
        raise ValueError(
            f"parameters must be a dict, got {type(params).__name__}"
        )
    if set(params) != set(func.parameters):
        raise ValueError(
            f"parameter keys do not match {func.name}: {set(params)}"
        )
    for key, param in func.parameters.items():
        value = params[key]
        if param.type == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{key}: number expects int|float, "
                    f"got {type(value).__name__}"
                )
        elif param.type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"{key}: string expects str, "
                    f"got {type(value).__name__}"
                )
        elif param.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{key}: boolean expects bool, "
                    f"got {type(value).__name__}"
                )
    return CallResult(
        prompt=str(obj["prompt"]), name=func.name, parameters=params
    )


def fallback_result(prompt: str, functions: list[FunctionDef]) -> CallResult:
    """Builds a schema-compliant result when model inference fails."""
    if not functions:
        return CallResult(prompt=prompt, name="", parameters={})
    function = functions[0]
    defaults: dict[str, object] = {}
    for key, parameter in function.parameters.items():
        if parameter.type == "number":
            defaults[key] = 0
        elif parameter.type == "boolean":
            defaults[key] = False
        else:
            defaults[key] = ""
    return CallResult(
        prompt=prompt, name=function.name, parameters=defaults
    )


def run(
    model: Small_LLM_Model, functions: list[FunctionDef],
    tests: list[PromptItem],
) -> list[CallResult]:
    """Runs everything in a concise pipeline in a try/except
    condition to avoid crash"""
    token_text = get_token_text(model)
    results: list[CallResult] = []
    for item in tests:
        out_of_memory = False
        try:
            base_ids = to_ids(model, build_prompt(functions, item.prompt))
            obj = generate_call(
                model, base_ids, item.prompt, functions, token_text
            )
            result = validate_result(obj, functions)
        except Exception as e:
            print(f"ERROR in prompt {item.prompt!r}: {e}", file=sys.stderr)
            result = fallback_result(item.prompt, functions)
            out_of_memory = "out of memory" in str(e).lower()
        results.append(result)
        if out_of_memory:
            remaining = tests[len(results):]
            results.extend(
                fallback_result(next_item.prompt, functions)
                for next_item in remaining
            )
            break
    return results


def write_output(output: str, results: list[CallResult]) -> None:
    """Saves the results as JSON, creating the output dir if missing."""
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)
