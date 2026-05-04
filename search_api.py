#!/usr/bin/env python3
"""Simple API for brute-force vector search over embedding JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/embeddings"
DEFAULT_MODEL = "intfloat/e5-mistral-7b-instruct"


@dataclass(frozen=True)
class SearchRecord:
    input_line: int
    dataset_id: str
    embedding_input: str


@dataclass
class EmbeddingIndex:
    path: Path
    model: str
    records: list[SearchRecord]
    matrix: np.ndarray


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    embeddings_file: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class SearchResult(BaseModel):
    rank: int
    score: float
    input_line: int
    dataset_id: str
    embedding_input: str


class SearchResponse(BaseModel):
    query: str
    embeddings_file: str
    embedding_model: str
    total_vectors: int
    results: list[SearchResult]


app = FastAPI(title="Metadata Embedding Search API")
index_cache: dict[tuple[str, int, int], EmbeddingIndex] = {}
index_cache_lock = threading.Lock()

embedding_endpoint = os.environ.get("EMBEDDING_ENDPOINT", DEFAULT_ENDPOINT)
embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)
embedding_timeout_seconds = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "120"))


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def load_embedding_index(path: Path) -> EmbeddingIndex:
    records: list[SearchRecord] = []
    embeddings: list[list[float]] = []
    model = ""

    for line_number, row in iter_jsonl(path):
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"{path}:{line_number}: expected non-empty embedding list")

        if not model:
            model = str(row.get("embedding_model") or embedding_model)

        records.append(
            SearchRecord(
                input_line=int(row.get("input_line") or line_number),
                dataset_id=str(row.get("dataset_id") or ""),
                embedding_input=str(row.get("embedding_input") or ""),
            )
        )
        embeddings.append(embedding)

    if not records:
        raise ValueError(f"{path}: no embedding rows found")

    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{path}: embeddings must form a 2D matrix")

    return EmbeddingIndex(
        path=path,
        model=model or embedding_model,
        records=records,
        matrix=normalize_rows(matrix),
    )


def get_embedding_index(path_string: str) -> EmbeddingIndex:
    path = Path(path_string).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    if not path.is_file():
        raise ValueError(f"{path} is not a file")

    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    with index_cache_lock:
        cached = index_cache.get(cache_key)
        if cached is not None:
            return cached

    loaded = load_embedding_index(path)
    with index_cache_lock:
        index_cache[cache_key] = loaded
    return loaded


def embed_query(query: str) -> np.ndarray:
    payload = {
        "model": embedding_model,
        "input": [f"query: {query}"],
        "encoding_format": "float",
    }
    response = httpx.post(embedding_endpoint, json=payload, timeout=embedding_timeout_seconds)
    response.raise_for_status()
    body = response.json()
    data = body.get("data", [])
    if len(data) != 1 or "embedding" not in data[0]:
        raise ValueError("embedding endpoint did not return exactly one embedding")
    return normalize_vector(np.asarray(data[0]["embedding"], dtype=np.float32))


def top_k(index: EmbeddingIndex, query_vector: np.ndarray, k: int) -> list[SearchResult]:
    if query_vector.shape[0] != index.matrix.shape[1]:
        raise ValueError(
            f"query dimension {query_vector.shape[0]} does not match "
            f"index dimension {index.matrix.shape[1]}"
        )

    scores = index.matrix @ query_vector
    result_count = min(k, scores.shape[0])
    candidate_indices = np.argpartition(scores, -result_count)[-result_count:]
    sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    results: list[SearchResult] = []
    for rank, index_position in enumerate(sorted_indices, start=1):
        record = index.records[int(index_position)]
        results.append(
            SearchResult(
                rank=rank,
                score=float(scores[index_position]),
                input_line=record.input_line,
                dataset_id=record.dataset_id,
                embedding_input=record.embedding_input,
            )
        )
    return results


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def search(request: SearchRequest) -> SearchResponse:
    try:
        index = get_embedding_index(request.embeddings_file)
        query_vector = embed_query(request.query.strip())
        results = top_k(index, query_vector, request.k)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"embedding endpoint failed: {exc}") from exc

    return SearchResponse(
        query=request.query,
        embeddings_file=str(index.path),
        embedding_model=index.model,
        total_vectors=len(index.records),
        results=results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a brute-force embedding search API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--endpoint", default=embedding_endpoint)
    parser.add_argument("--model", default=embedding_model)
    parser.add_argument("--timeout-seconds", type=float, default=embedding_timeout_seconds)
    return parser.parse_args()


def main() -> None:
    global embedding_endpoint, embedding_model, embedding_timeout_seconds

    args = parse_args()
    embedding_endpoint = args.endpoint
    embedding_model = args.model
    embedding_timeout_seconds = args.timeout_seconds
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
