#!/usr/bin/env python3
"""Prepare DCAT JSON-LD metadata for embeddings and call a local embedding API."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/embeddings"
DEFAULT_MODEL = "intfloat/e5-mistral-7b-instruct"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def scalar_values(value: Any) -> list[str]:
    """Extract human-readable values from JSON-LD scalar/list objects."""
    values: list[str] = []
    for item in as_list(value):
        if item is None:
            continue
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            if "@value" in item:
                values.extend(scalar_values(item["@value"]))
            elif "rdfs:label" in item:
                values.extend(scalar_values(item["rdfs:label"]))
            elif "@id" in item:
                values.append(str(item["@id"]))
        else:
            values.append(str(item))
    return [clean_text(v) for v in values if clean_text(v)]


def first_scalar(value: Any) -> str:
    values = scalar_values(value)
    return values[0] if values else ""


def clean_text(value: str) -> str:
    return " ".join(value.split())


def node_types(node: dict[str, Any]) -> set[str]:
    return set(str(value) for value in as_list(node.get("@type")))


def ref_ids(value: Any) -> list[str]:
    refs: list[str] = []
    for item in as_list(value):
        if isinstance(item, dict) and "@id" in item:
            refs.append(str(item["@id"]))
        elif isinstance(item, str):
            refs.append(item)
    return refs


def find_dataset_node(graph: list[dict[str, Any]]) -> dict[str, Any] | None:
    for node in graph:
        if "dcat:Dataset" in node_types(node):
            return node
    return None


def extract_embedding_input(raw: dict[str, Any]) -> str:
    graph = [node for node in as_list(raw.get("@graph")) if isinstance(node, dict)]
    nodes_by_id = {str(node["@id"]): node for node in graph if "@id" in node}
    dataset = find_dataset_node(graph)
    if dataset is None:
        raise ValueError("row does not contain a dcat:Dataset node")

    distribution_titles: list[str] = []
    for distribution_id in ref_ids(dataset.get("dcat:distribution")):
        distribution = nodes_by_id.get(distribution_id)
        if distribution:
            distribution_titles.extend(scalar_values(distribution.get("dct:title")))

    title = first_scalar(dataset.get("dct:title"))
    description = first_scalar(dataset.get("dct:description"))
    keywords = scalar_values(dataset.get("dcat:keyword"))

    embedding_fields = {
        "dct:title": title,
        "dct:description": description,
        "dcat:keyword": keywords,
        "distribution_dct:title": distribution_titles,
    }
    return build_embedding_input(embedding_fields)


def build_embedding_input(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record["dct:title"]:
        parts.append(f"Title: {record['dct:title']}")
    if record["dct:description"]:
        parts.append(f"Description: {record['dct:description']}")
    if record["dcat:keyword"]:
        parts.append(f"Keywords: {', '.join(record['dcat:keyword'])}")
    if record["distribution_dct:title"]:
        parts.append(f"Distribution titles: {', '.join(record['distribution_dct:title'])}")
    return "passage: " + "\n".join(parts)


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def write_prepared_jsonl(input_path: Path, output_path: Path) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for _, raw in iter_jsonl(input_path):
            embedding_input = extract_embedding_input(raw)
            output.write(json.dumps(embedding_input, ensure_ascii=False) + "\n")
            count += 1
    return count


def chunks(items: list[str], size: int) -> Any:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def read_embedding_inputs(path: Path) -> list[str]:
    inputs: list[str] = []
    for line_number, raw in iter_jsonl(path):
        if not isinstance(raw, str):
            raise ValueError(
                f"{path}:{line_number}: expected a JSON string containing the embedding input"
            )
        inputs.append(raw)
    return inputs


def fetch_embeddings(
    client: Any,
    endpoint: str,
    model: str,
    texts: list[str],
    timeout_seconds: float,
    retries: int,
) -> list[list[float]]:
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            response = client.post(endpoint, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            body = response.json()
            data = body.get("data", [])
            if len(data) != len(texts):
                raise ValueError(f"expected {len(texts)} embeddings, got {len(data)}")
            return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
        except Exception as exc:
            last_error = exc
            if attempt <= retries:
                time.sleep(min(2**attempt, 30))
                continue
            raise RuntimeError(f"embedding request failed after {attempt} attempt(s): {exc}") from exc
    raise RuntimeError(f"embedding request failed: {last_error}")


def write_embeddings_jsonl(
    prepared_path: Path,
    output_path: Path,
    endpoint: str,
    model: str,
    batch_size: int,
    timeout_seconds: float,
    retries: int,
) -> int:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "httpx is required for embedding calls. Install dependencies with `uv sync` "
            "or run this script with `uv run python prepare_and_embed_metadata.py ...`."
        ) from exc

    embedding_inputs = read_embedding_inputs(prepared_path)
    count = 0
    with httpx.Client(headers={"Content-Type": "application/json"}) as client:
        with output_path.open("w", encoding="utf-8") as output:
            for batch in chunks(embedding_inputs, batch_size):
                embeddings = fetch_embeddings(
                    client=client,
                    endpoint=endpoint,
                    model=model,
                    texts=batch,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                )
                for embedding_input, embedding in zip(batch, embeddings, strict=True):
                    embedded_record = {
                        "input_line": count + 1,
                        "embedding_input": embedding_input,
                        "embedding_model": model,
                        "embedding": embedding,
                    }
                    output.write(json.dumps(embedded_record, ensure_ascii=False) + "\n")
                    count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare DCAT JSON-LD metadata JSONL and create local embeddings JSONL."
    )
    parser.add_argument("input_jsonl", type=Path, help="Input JSONL, one metadata entry per row.")
    parser.add_argument(
        "--prepared-output",
        type=Path,
        default=Path("metadata_for_embeddings.jsonl"),
        help="Output JSONL containing one embedding input string per row.",
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=Path("metadata_with_embeddings.jsonl"),
        help="Output JSONL containing embedding input strings plus embedding vectors.",
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Embedding API endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model name.")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding request batch size.")
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="HTTP request timeout.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per embedding batch.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the prepared JSONL; do not call the embedding API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    prepared_count = write_prepared_jsonl(args.input_jsonl, args.prepared_output)
    print(f"wrote {prepared_count} embedding input rows to {args.prepared_output}", file=sys.stderr)

    if args.prepare_only:
        return 0

    embedded_count = write_embeddings_jsonl(
        prepared_path=args.prepared_output,
        output_path=args.embeddings_output,
        endpoint=args.endpoint,
        model=args.model,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )
    print(f"wrote {embedded_count} embedded records to {args.embeddings_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
