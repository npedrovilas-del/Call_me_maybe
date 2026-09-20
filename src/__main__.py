import argparse, sys

from llm_sdk import Small_LLM_Model
from src.loader import load_functions, load_tests
from src.validation import run, write_output
from src.models import CallResult

def parse_args() -> argparse.Namespace:
    """Parser of the terminal code """
    parser = argparse.ArgumentParser(description="Constrained function calling (Marco 7).")
    parser.add_argument("--functions_definition", default="data/input/functions_definitions.json")
    parser.add_argument("--input", default="data/input/function_calling_tests.json")
    parser.add_argument("--output", default="data/output/function_calling_results.json")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    try:
        functions = load_functions(args.functions_definition)
        tests = load_tests(args.input)
        model = Small_LLM_Model()
        results = run(model, functions, tests)
        write_output(args.output, results)
    except Exception as e:
        print(f"Fatal Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
