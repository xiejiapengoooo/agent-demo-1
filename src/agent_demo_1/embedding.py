import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from http import HTTPStatus
from typing import Any

import dashscope

EMBEDDING_MODEL = "tongyi-embedding-vision-plus"
BATCH_SIZE = 10
MAX_WORKERS = 3
MAX_RETRIES = 3
RETRY_DELAY = 2


def embed_chunks(
    chunks: Sequence[Mapping[str, Any]],
) -> list[list[float]]:
    inputs = [_embedding_input(chunk, index) for index, chunk in enumerate(chunks)]
    if not inputs:
        return []

    batches = [
        (start, inputs[start : start + BATCH_SIZE])
        for start in range(0, len(inputs), BATCH_SIZE)
    ]
    print(
        f"Embedding: 总共 {len(chunks)} 个 chunk，分成 {len(batches)} 批，"
        f"每批最多 {BATCH_SIZE} 个"
    )

    ordered_results: list[list[float] | None] = [None] * len(inputs)
    workers = min(MAX_WORKERS, len(batches))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[list[list[float]]], tuple[int, int]] = {}
        for batch_index, (start, batch) in enumerate(batches):
            future = executor.submit(_embed_batch, batch, batch_index)
            futures[future] = (start, len(batch))

        try:
            for future in as_completed(futures):
                start, batch_length = futures[future]
                vectors = future.result()
                if len(vectors) != batch_length:
                    raise RuntimeError(
                        f"embedding batch returned {len(vectors)} vectors; "
                        f"expected {batch_length}"
                    )
                ordered_results[start : start + batch_length] = vectors
        except Exception:
            for future in futures:
                future.cancel()
            raise

    if any(vector is None for vector in ordered_results):
        raise RuntimeError("embedding results are incomplete")

    results = [vector for vector in ordered_results if vector is not None]
    dimensions = {len(vector) for vector in results}
    if len(dimensions) != 1:
        raise RuntimeError(f"embedding dimensions are inconsistent: {dimensions}")
    return results


def _embedding_input(chunk: Mapping[str, Any], chunk_index: int) -> dict[str, str]:
    if not isinstance(chunk, Mapping):
        raise TypeError(f"chunk at index {chunk_index} must be a mapping")

    chunk_type = chunk.get("type")
    if chunk_type == "image":
        source = chunk.get("source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"image chunk at index {chunk_index} has no source")
        return {"image": source.strip()}

    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk at index {chunk_index} has no text")
    return {"text": text.strip()}


def _embed_batch(
    batch: Sequence[Mapping[str, str]],
    batch_index: int,
) -> list[list[float]]:
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            response = dashscope.MultiModalEmbedding.call(
                model=EMBEDDING_MODEL,
                input=list(batch),
            )
            status_code = _response_value(response, "status_code")
            if status_code != HTTPStatus.OK:
                code = _response_value(response, "code", "unknown")
                message = _response_value(response, "message", "unknown error")
                raise RuntimeError(
                    f"DashScope request failed with status {status_code}: "
                    f"{code} - {message}"
                )

            vectors = _response_vectors(response, len(batch))
            print(f"Embedding: 批次 {batch_index + 1} 成功，返回 {len(vectors)} 个向量")
            return vectors
        except Exception as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2**attempt)
                print(
                    f"Embedding: 批次 {batch_index + 1} 失败，"
                    f"{delay} 秒后进行第 {attempt + 2}/{MAX_RETRIES} 次尝试：{error}"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"embedding batch {batch_index + 1} failed after {MAX_RETRIES} attempts"
    ) from last_error


def _response_vectors(response: Any, expected_count: int) -> list[list[float]]:
    output = _response_value(response, "output")
    if not isinstance(output, Mapping):
        raise TypeError("DashScope response has no output object")

    embeddings = output.get("embeddings")
    if not isinstance(embeddings, Sequence) or isinstance(embeddings, (str, bytes)):
        raise TypeError("DashScope response has no embeddings list")
    if len(embeddings) != expected_count:
        raise ValueError(
            f"DashScope returned {len(embeddings)} embeddings; "
            f"expected {expected_count}"
        )

    vectors: list[list[float] | None] = [None] * expected_count
    for result in embeddings:
        if not isinstance(result, Mapping):
            raise TypeError("DashScope returned an invalid embedding item")

        result_index = result.get("index")
        if (
            not isinstance(result_index, int)
            or isinstance(result_index, bool)
            or not 0 <= result_index < expected_count
        ):
            raise ValueError(
                f"DashScope returned invalid embedding index: {result_index!r}"
            )
        if vectors[result_index] is not None:
            raise ValueError(
                f"DashScope returned duplicate embedding index: {result_index}"
            )

        embedding = result.get("embedding")
        if not isinstance(embedding, Sequence) or isinstance(embedding, (str, bytes)):
            raise TypeError(f"embedding at index {result_index} is not a vector")
        if not embedding:
            raise ValueError(f"embedding at index {result_index} is empty")

        vector = []
        for value in embedding:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(
                    f"embedding at index {result_index} contains a non-numeric value"
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(
                    f"embedding at index {result_index} contains a non-finite value"
                )
            vector.append(numeric_value)
        vectors[result_index] = vector

    if any(vector is None for vector in vectors):
        raise ValueError("DashScope response is missing an embedding index")
    return [vector for vector in vectors if vector is not None]


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    if isinstance(response, Mapping):
        return response.get(key, default)
    return getattr(response, key, default)


__all__ = ["embed_chunks"]
