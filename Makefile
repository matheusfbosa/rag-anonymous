docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-clean:
	docker compose down -v
	docker compose rm -f
	docker compose pull
	docker compose up -d

docker-ollama-pull-models:
	docker exec -it ollama ollama pull nomic-embed-text
	docker exec -it ollama ollama pull qwen3:0.6b
	docker exec -it ollama ollama pull qwen3:1.7b
	docker exec -it ollama ollama pull qwen3:4b
	docker exec -it ollama ollama pull gemma4:e2b
	docker exec -it ollama ollama pull gemma4:e4b

ollama-pull-models:
	ollama pull nomic-embed-text
	ollama pull qwen3:0.6b
	ollama pull qwen3:1.7b
	ollama pull qwen3:4b
	ollama pull gemma4:e2b
	ollama pull gemma4:e4b

ingest:
	rag-anon ingest

query:
	rag-anon query

test:
	pytest -q
