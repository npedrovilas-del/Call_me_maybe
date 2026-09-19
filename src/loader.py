import json
import sys

from pydantic import ValidationError

from src.models import FunctionDef, PromptItem


def load_functions(path: str) -> list[FunctionDef]:
    """Loads the functions JSON and validates if its valid 
    based on the parameters i put in the models classes if 
    it is return a list already validate and integrated in the
    FunctionDef class """
    with open(path, "r") as f:
        data = json.load(f) # json.load turns the json file into a dictionary of  dictonaryes in like prompt: dada value: 313 3131 3131
    return [FunctionDef.model_validate(x) for x in data]


def load_tests(path: str) -> list[PromptItem]:
    """Loads the tests JSON and validates if its valid based on 
    the parameters i put in the models classes if it 
    is return a list already validate and integrated in the
    PromptItem class """
    with open(path, "r") as f:
        data = json.load(f)
    return [PromptItem.model_validate(x) for x in data]


def safe_test(path: str) -> list[PromptItem]:
    """ Makes the errors that can occur on the test files be much 
    more easy to fix and doesnt crash the program 
    it closes it very gracefull"""
    try:
        f = load_tests(path)
        return f
    except ValidationError as e:
        print("The parameters of the input are not valid", e)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("JSON File Broken:", e)
        sys.exit(1)
    except FileNotFoundError as e:
        print("JSON File Not found:", e)
        sys.exit(1)


def safe_definitions(path: str) -> list[FunctionDef]:
    """ Makes the errors that can occur on the definition 
    files be much more easy to fix and doesnt crash the program 
    it closes it very gracefull"""
    try:
        f = load_functions(path)
        return f
    except ValidationError as e:
        print("The parameters of the input are not valid", e)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print("JSON File Broken:", e)
        sys.exit(1)
    except FileNotFoundError as e:
        print("JSON File Not found:", e)
        sys.exit(1)