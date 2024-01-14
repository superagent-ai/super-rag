format:
	python -m black .
	python -m ruff --select I --fix .

PYTHON_FILES=.
lint: PYTHON_FILES=.
lint_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$')

lint lint_diff:
	python -m black $(PYTHON_FILES) --check
	python -m ruff .