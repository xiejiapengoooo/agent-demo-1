import json
from pathlib import Path

from .embedding import EMBEDDING_MODEL
from .persisting import persist_chunks


def build_index():
    # parser_registry = ParserRegistry()
    # parser_registry.register(DocxParser)

    # file = Path(
    #     "source/迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx"
    # )

    # result = parser_registry.parse(file)
    # if not result or not result.get("output"):
    #     raise ValueError("no output")

    # data = result.get("output")

    # blocks = normalize_blocks(data)

    # chunks = chunk_blocks(
    #     blocks,
    #     document_id="迪士尼乐园度假区客诉处理流程，不同类型投诉（服务态度、设施故障、商品瑕疵）的应对话术与补偿方案模板.docx",
    # )
    #
    # embedded_chunks = embed_chunks(chunks)

    with Path("mock_1.json").open(encoding="utf-8") as data_file:
        embedded_chunks = json.load(data_file)

    manifest = persist_chunks(
        embedded_chunks,
        embedding_model=EMBEDDING_MODEL,
    )
    print(
        f"Persisted {manifest['chunk_count']} chunks to "
        f"data/{manifest['files']['sqlite']} and data/{manifest['files']['faiss']}"
    )


__all__ = ["build_index"]
