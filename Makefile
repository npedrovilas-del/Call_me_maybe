export UV_CACHE_DIR := /goinfre/$(USER)/uv-cache
export HF_HOME := /goinfre/$(USER)/hf-cache

install:
	mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	uv sync

run:
	mkdir -p $(UV_CACHE_DIR) $(HF_HOME)
	uv run python -m src \
		--functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json
debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache src/__pycache__

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	      --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 .
	mypy . --strict
