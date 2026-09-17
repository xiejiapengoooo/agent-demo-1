from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
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
):
    batches = [chunks[i : i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]
    print(
        f"Embedding: 总共 {len(chunks)} 个 chunk，分成 {len(batches)} 批，每批最多 {BATCH_SIZE} 个"
    )

    all_results = []

    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     future_to_batch = {
    #         executor.submit(embed_batch, batch, idx): idx
    #         for idx, batch in enumerate(batches)
    #     }

    #     for future in as_completed(future_to_batch):
    #         batch_idx = future_to_batch[future]
    #         try:
    #             results = future.result()
    #             all_results.extend(results)
    #         except Exception as e:
    #             print(f"批次 {batch_idx} 处理异常: {e}")

    # return all_results


# def embed_batch(batch: list[dict[str, Any]], batch_idx: int):
#     for attempt in range(MAX_RETRIES):
#         try:
#             resp = dashscope.MultiModalEmbedding.call(
#                 model=EMBEDDING_MODEL, input=batch
#             )

#             if resp.status_code == HTTPStatus.OK:
#                 embeddings = resp.output.get("embeddings", [])
#                 results = []
#                 for emb in embeddings:
#                     results.append(
#                         {
#                             "index": emb.get("index"),
#                             "type": emb.get("type"),
#                             "embedding": emb.get("embedding"),
#                             # 可以在这里把原始 chunk 信息也带回去
#                         }
#                     )
#                 print(f"✅ 批次 {batch_idx} 成功，返回 {len(results)} 个向量")
#                 return results
#             else:
#                 print(
#                     f"❌ 批次 {batch_idx} 失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {resp.code} - {resp.message}"
#                 )

#         except Exception as e:
#             print(
#                 f"❌ 批次 {batch_idx} 异常 (尝试 {attempt + 1}/{MAX_RETRIES}): {str(e)}"
#             )

#         if attempt < MAX_RETRIES - 1:
#             time.sleep(RETRY_DELAY * (attempt + 1))  # 指数退避

#     print(f"💥 批次 {batch_idx} 最终失败")
#     return []


__all__ = ["embed_chunks"]
