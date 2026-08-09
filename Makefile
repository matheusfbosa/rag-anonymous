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

ingest:
	rag-anon ingest

query:
	rag-anon query

test:
	pytest -q
