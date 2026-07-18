.PHONY: setup api test lint db

setup:
	uv sync
	cp -n .env.example .env || true

api:
	uv run uvicorn api.main:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

db:
	@echo "Run schema/migrations/001_init.sql against Supabase (SQL editor or psql \$$DATABASE_URL)"
