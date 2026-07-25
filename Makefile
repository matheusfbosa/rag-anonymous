docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-clean:
	docker compose down -v
	docker compose rm -f
	docker compose pull
	docker compose up -d

docker-ollama-up:
	docker compose -f docker-compose.ollama.yml up -d

docker-ollama-down:
	docker compose -f docker-compose.ollama.yml down

docker-ollama-pull-models:
	docker exec -it ollama ollama pull nomic-embed-text
	docker exec -it ollama ollama pull qwen3:8b
	docker exec -it ollama ollama pull gemma4:12b

ollama-pull-models:
	ollama pull nomic-embed-text
	ollama pull qwen3:8b
	ollama pull gemma4:12b

ingest:
	rag-anon ingest

query:
	rag-anon query

test:
	pytest -q
