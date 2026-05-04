# embeddings_hackaton

Prepare dataset metadata embeddings:

```bash
uv run python prepare_and_embed_metadata.py example_data_first_100.jsonl \
  --prepared-output metadata_for_embeddings.jsonl \
  --embeddings-output metadata_with_embeddings.jsonl
```

Run the brute-force search API:

```bash
uv run python search_api.py --host 0.0.0.0 --port 8080
```

Search an embeddings JSONL file:

```bash
curl http://127.0.0.1:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "point cloud registration methods",
    "embeddings_file": "metadata_with_embeddings.jsonl",
    "k": 5
  }'
```
