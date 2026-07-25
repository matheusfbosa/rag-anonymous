# rag-anonymous

RAG pipeline with Offline/On-Demand anonymization.

## Architecture

The pipeline supports two anonymization strategies controlled by `RAG_ANON_ANONYMIZER_STRATEGY`.
In **Offline** mode, PII is removed from the corpus before indexing so vectors and retrieved context are always redacted.
In **On-Demand** mode, raw text is indexed and only the final LLM answer is anonymized at query time.

```mermaid
flowchart TB
    subgraph ingest_phase ["Ingest  (rag-anon ingest)"]
        direction TB
        corpus["Corpus\n(RAG_ANON_CORPUS)"] --> load["Download / Load"]
        load --> strat_in{"Strategy?"}
        strat_in -->|offline| presidio_in["Presidio Anonymizer\n(replaces PII with tags)"]
        presidio_in --> splitter_off["Text Splitter\n(RecursiveCharacter)"]
        strat_in -->|ondemand| splitter_on["Text Splitter\n(RecursiveCharacter)"]
        splitter_off --> embed["Ollama Embeddings\n(RAG_ANON_EMBEDDING_MODEL)"]
        splitter_on --> embed
        embed --> es[("Elasticsearch\n(offline or ondemand index)")]
    end

    subgraph query_phase ["Query  (rag-anon query)"]
        direction TB
        question["User Question"] --> retriever["Elasticsearch Retriever\n(top-k chunks)"]
        retriever --> prompt_builder["Prompt Builder\n(context + question)"]
        prompt_builder --> llm["ChatOllama LLM\n(RAG_ANON_LLM_MODEL)"]
        llm --> strat_out{"Strategy?"}
        strat_out -->|offline| answer["Answer"]
        strat_out -->|ondemand| presidio_out["Presidio Anonymizer\n(scrubs LLM output)"]
        presidio_out --> answer
    end

    es --> retriever
```

## Project structure

```
rag-anonymous/
├── pyproject.toml          # Package metadata, dependencies, rag-anon entry point
├── rag_anonymous/
│   ├── __init__.py
│   ├── config.py           # Load env vars
│   ├── cli.py              # rag-anon ingest|query
│   ├── anonymizer.py       # Presidio-based PII anonymization
│   ├── ingest.py           # Corpus download, anonymization, chunking, indexing
│   ├── log_config.py       # Logging setup
│   └── query.py            # RAG chain and retrieval
├── data/                   # Cached corpus downloads (data/input/corpus/)
├── docker-compose.yml         # Elasticsearch + Kibana
├── docker-compose.ollama.yml  # Elasticsearch + Kibana + Ollama
├── Makefile
└── .env.example
```

## Setup

### 1. Ollama

The pipeline talks to Ollama over its HTTP API at `RAG_ANON_OLLAMA_BASE_URL` (default `http://localhost:11434`). Two equivalent ways to bring it up — pick one:

**Option A — Native CLI / desktop app** (install from [https://ollama.com](https://ollama.com)):

```sh
ollama serve                      # or just launch the Ollama desktop app
make ollama-pull-models           # pull models via the host CLI
make docker-up                    # start Elasticsearch + Kibana (Ollama runs natively)
```

Both paths expose the same API on port `11434`, so the rest of the pipeline doesn't need to know which one is running. **Don't run both at once** — they'll race for the port and the second one will fail to bind.

**Option B — Docker** (uses `[docker-compose.ollama.yml](./docker-compose.ollama.yml)` — Ollama alongside Elasticsearch + Kibana):

```sh
make docker-ollama-up             # start Ollama + Elasticsearch + Kibana
make docker-ollama-pull-models    # pull models inside the container
make docker-ollama-down           # stop the containers when done
```

### 2. Python package

```sh
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .

cp .env.example .env
```

`pip install -e .` reads `pyproject.toml` and installs the `rag_anonymous` package locally in editable mode.

## Usage

The CLI takes no flags — all knobs come from environment variables (typically `.env`). Override individual variables inline when needed.

### Ingest corpus

- **Offline:** Anonymizes the corpus before indexing (PII replaced by tags in stored chunks).
- **On-Demand:** Indexes the corpus as-is (raw text); anonymization is applied to the LLM answer at query time.

Reads `RAG_ANON_ANONYMIZER_STRATEGY` and `RAG_ANON_CORPUS_DATASET` from env:

```sh
make ingest
# or override inline:
RAG_ANON_ANONYMIZER_STRATEGY=ondemand RAG_ANON_CORPUS_DATASET=train make ingest
```

Equivalent direct CLI:

```sh
rag-anon ingest
```

### Query

Reads `RAG_ANON_ANONYMIZER_STRATEGY`, `RAG_ANON_RETRIEVAL_K_DOCS`, and `RAG_ANON_QUERY_QUESTION` from env. Use the same strategy as at ingest time so the correct index is queried. For `ondemand`, the generated answer is anonymized before being returned.

```sh
RAG_ANON_QUERY_QUESTION="What did the applicant complain about in case no. 13146/02?" make query
# override more variables:
RAG_ANON_ANONYMIZER_STRATEGY=ondemand RAG_ANON_RETRIEVAL_K_DOCS=10 RAG_ANON_QUERY_QUESTION="..." make query
```

Equivalent direct CLI:

```sh
RAG_ANON_QUERY_QUESTION="..." rag-anon query
```

## Configuration


| Variable                        | Default                     | Description                                                                                                                                                                                                                      |
| ------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RAG_ANON_ANONYMIZER_ENTITIES`  | `PERSON,DATE_TIME,LOCATION` | Presidio entity types                                                                                                                                                                                                            |
| `RAG_ANON_ANONYMIZER_STRATEGY`  | `offline`                   | Anonymization strategy (`offline` or `ondemand`)                                                                                                                                                                                 |
| `RAG_ANON_CHUNK_OVERLAP`        | `0`                         | Chunk overlap, in tokens (`cl100k_base`)                                                                                                                                                                                         |
| `RAG_ANON_CHUNK_SIZE`           | `200`                       | Chunk size, in tokens (`cl100k_base`)                                                                                                                                                                                            |
| `RAG_ANON_CORPUS`               | TAB ECHR URL template       | Corpus source as a URL or local path, with optional `{dataset}`. `http(s)://` is downloaded and cached; anything else is read as a local file. Each item needs `text` and `doc_id` fields.                                       |
| `RAG_ANON_CORPUS_DATASET`       | `dev`                       | Dataset name interpolated into `{dataset}` in `RAG_ANON_CORPUS` (default TAB datasets: train/dev/test)                                                                                                                           |
| `RAG_ANON_EMBEDDING_MODEL`      | `nomic-embed-text`          | Embeddings model                                                                                                                                                                                                                 |
| `RAG_ANON_EMBEDDING_NUM_CTX`    | `8192`                      | Ollama context window for the embedding model, in tokens. Used at ingest time to embed each chunk; must be ≥ `RAG_ANON_CHUNK_SIZE` or chunks are silently truncated before embedding. Distinct from `RAG_ANON_LLM_NUM_CTX`.       |
| `RAG_ANON_ES_URL`               | `http://localhost:9200`     | Elasticsearch base URL                                                                                                                                                                                                           |
| `RAG_ANON_LLM_MODEL`            | `qwen3:8b`                  | Generator model                                                                                                                                                                                                                  |
| `RAG_ANON_LLM_NUM_CTX`          | `8192`                      | Ollama context window size in tokens. Caps the total tokens (prompt + generated output) sent to the model. Increase if retrieved chunks + prompt exceed the default; lower to reduce VRAM usage.                                 |
| `RAG_ANON_LLM_REASONING`        | `false`                     | Maps to `ChatOllama(reasoning=...)`. `false` disables thinking via Ollama's `think` flag (faster, cleaner output). Set to `true` only when running a reasoning-capable model and you want `<think>` content captured separately. |
| `RAG_ANON_LLM_TEMPERATURE`      | `0.0`                       | Generation temperature                                                                                                                                                                                                           |
| `RAG_ANON_LOG_LEVEL`            | `INFO`                      | Application loggers / root level. Accepts `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL` or a numeric level. Also picked up by `rag-anonymous-metrics`.                                                                             |
| `RAG_ANON_LOG_LEVEL_HTTP`       | `WARNING`                   | Level for the HTTP client loggers (`httpx`, `httpcore`, `urllib3`). Set to `INFO` to see every Ollama HTTP call.                                                                                                                 |
| `RAG_ANON_LOG_LEVEL_PRESIDIO`   | `ERROR`                     | Level for `presidio-analyzer` / `presidio-anonymizer` loggers. The default suppresses the noisy startup banner; raise to `INFO` to debug recognizer issues.                                                                      |
| `RAG_ANON_OLLAMA_BASE_URL`      | `http://localhost:11434`    | Ollama API endpoint                                                                                                                                                                                                              |
| `RAG_ANON_QUERY_QUESTION`       | `""`                        | Question used by `rag-anon query`                                                                                                                                                                                                |
| `RAG_ANON_RETRIEVAL_K_DOCS`     | `3`                         | Retrieved chunks per query                                                                                                                                                                                                       |
