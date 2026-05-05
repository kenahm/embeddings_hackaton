# Embeddings Hackathon

Prototype workflow for embedding Austrian metadata JSONL files and searching them with a small brute-force API.

## Setup

Install the Python environment with `uv`:

```bash
uv sync
```

The embedding model server must already be running and must expose an OpenAI-compatible embeddings endpoint at:

```text
http://127.0.0.1:8000/v1/embeddings
```

The default model name used by the scripts is:

```text
intfloat/e5-mistral-7b-instruct
```

## Build The Wien Prototype Database

The small prototype input file is:

```text
katalog_metadaten_total_wien.jsonl
```

Create the embedding input rows and the final vector database:

```bash
uv run python prepare_and_embed_metadata.py katalog_metadaten_total_wien.jsonl \
  --prepared-output metadata_for_embeddings_wien.jsonl \
  --embeddings-output metadata_with_embeddings_wien.jsonl
```

This writes two generated files:

```text
metadata_for_embeddings_wien.jsonl
metadata_with_embeddings_wien.jsonl
```

`metadata_for_embeddings_wien.jsonl` contains exactly the strings sent to the embedding model, one JSON string per row.

`metadata_with_embeddings_wien.jsonl` contains the same row identity plus vectors, including `dataset_id`, `title`/`description` source text, model name, and the embedding vector.

## Run The Search API

Start the API server:

```bash
uv run python search_api.py --host 0.0.0.0 --port 8080
```

The API loads `metadata_with_embeddings_wien.jsonl` on the first query, converts vectors into a normalized NumPy matrix, and keeps that matrix cached in memory while the server is running.

## Query The Wien Database

Human-readable output:

```bash
curl http://127.0.0.1:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Radwege in Wien",
    "embeddings_file": "metadata_with_embeddings_wien.jsonl"
  }'
```

Ask for a specific number of results:

```bash
curl http://127.0.0.1:8080/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "öffentliche Verkehrsmittel Haltestellen",
    "embeddings_file": "metadata_with_embeddings_wien.jsonl",
    "k": 5
  }'
```

JSON output for use from another program:

```bash
curl http://127.0.0.1:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Schulen und Bildungseinrichtungen",
    "embeddings_file": "metadata_with_embeddings_wien.jsonl",
    "k": 5
  }'
```

Each result includes:

```text
dataset_id
title
description
score
input_line
```

## Notes

Stored dataset/document embeddings use the `passage:` prefix.

Search queries use the `query:` prefix internally.

Generated embedding output files are ignored by Git via `.gitignore`.
