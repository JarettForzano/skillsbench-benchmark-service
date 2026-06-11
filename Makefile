.PHONY: test lint serve

test:
	uv run pytest

lint:
	uv run ruff check src tests scripts

serve:
	uv run uvicorn skillsbench_valkyrie.main:app --host 127.0.0.1 --port 8001
