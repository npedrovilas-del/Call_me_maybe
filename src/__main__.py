import argparse
import sys

from llm_sdk import Small_LLM_Model
from src.loader import safe_definitions
from src.loader import safe_test
from src.validation import run
from src.validation import write_output


def parse_args() -> argparse.Namespace:
    """Parser of the terminal code """
    parser = argparse.ArgumentParser(
        description="Constrained function calling (Marco 7)."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definitions.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        functions = safe_definitions(args.functions_definition)
        tests = safe_test(args.input)
        model = Small_LLM_Model()
        results = run(model, functions, tests)
        write_output(args.output, results)
    except Exception as e:
        print(f"Fatal Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
