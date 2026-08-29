DATASET  ?= dev
STRATEGY ?= offline

# Lint / format

.PHONY: format lint test hooks
format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .
	ruff format --check .
	mypy

hooks:
	pre-commit install

# Test

test:
	pytest -q

# Docker

docker-up:
	docker compose up -d

docker-up-ollama:
	docker compose --profile ollama up -d

docker-up-kibana:
	docker compose --profile kibana up -d kibana

docker-up-all:
	docker compose --profile ollama --profile kibana up -d

docker-ollama-down:
	docker compose stop ollama

docker-kibana-down:
	docker compose stop kibana

docker-down:
	docker compose down

docker-clean:
	docker compose down -v
	docker compose rm -f
	docker compose pull
	docker compose up -d

# Ollama models

docker-ollama-pull-models:
	docker exec -it ollama ollama pull nomic-embed-text
	docker exec -it ollama ollama pull qwen3:4b
	docker exec -it ollama ollama pull qwen3:8b
	docker exec -it ollama ollama pull gemma4:12b

ollama-pull-models:
	ollama pull nomic-embed-text
	ollama pull qwen3:4b
	ollama pull qwen3:8b
	ollama pull gemma4:12b

# Ingest

ingest:
	RAG_ANON_ANONYMIZER_STRATEGY=$(STRATEGY) \
	RAG_ANON_CORPUS_DATASET=$(DATASET) \
	rag-anon ingest

ingest-offline:
	$(MAKE) ingest STRATEGY=offline DATASET=$(DATASET)

ingest-ondemand:
	$(MAKE) ingest STRATEGY=ondemand DATASET=$(DATASET)

ingest-dev:
	$(MAKE) ingest DATASET=dev

ingest-train:
	$(MAKE) ingest DATASET=train

ingest-test:
	$(MAKE) ingest DATASET=test

ingest-strategies:
	$(MAKE) ingest-offline DATASET=$(DATASET)
	$(MAKE) ingest-ondemand DATASET=$(DATASET)

ingest-dev-strategies:
	$(MAKE) ingest-strategies DATASET=dev

ingest-train-strategies:
	$(MAKE) ingest-strategies DATASET=train

ingest-test-strategies:
	$(MAKE) ingest-strategies DATASET=test

ingest-all:
	$(MAKE) ingest-dev-strategies
	$(MAKE) ingest-train-strategies
	$(MAKE) ingest-test-strategies

# Query

query:
	rag-anon query
