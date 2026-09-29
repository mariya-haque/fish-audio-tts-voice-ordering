"""Retrieval over the local Chroma knowledge base.

Follows the shape of RAG/rag_engine.py's retrieve_context / build_context_block,
but against this app's own `voice_kb` collection so the voice agent is not tied
to that project's corpus.

Latency note: the query embedding is a network round trip sitting directly on the
critical path of every turn (~100-200ms). retrieve() therefore short-circuits
when RAG is disabled or the index is empty, so an unpopulated KB costs nothing.
"""

import asyncio
import logging
from functools import lru_cache

import chromadb
from openai import AsyncOpenAI

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _collection():
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(config.COLLECTION_NAME)


@lru_cache(maxsize=1)
def _openai() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=config.OPENAI_API_KEY)


def collection_count() -> int:
    try:
        return _collection().count()
    except Exception as exc:
        logger.warning("chroma unavailable: %s", exc)
        return 0


async def embed(texts: list[str]) -> list[list[float]]:
    response = await _openai().embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


async def retrieve(query: str, top_k: int | None = None) -> list[tuple[str, dict]]:
    """Return [(chunk_text, metadata)] most relevant to `query`, or [] if unavailable."""
    if not config.RAG_ENABLED or not query.strip():
        return []

    count = await asyncio.to_thread(collection_count)
    if count == 0:
        return []

    try:
        [query_embedding] = await embed([query])
        results = await asyncio.to_thread(
            lambda: _collection().query(
                query_embeddings=[query_embedding],
                n_results=min(top_k or config.RAG_TOP_K, count),
            )
        )
    except Exception as exc:
        # A retrieval failure should degrade to an ungrounded answer, never kill the turn.
        logger.warning("retrieval failed, continuing without context: %s", exc)
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0] or [{}] * len(docs)
    return list(zip(docs, metas))


def build_context_block(retrieved: list[tuple[str, dict]]) -> str:
    """Render retrieved chunks into a single prompt-ready block."""
    blocks = []
    for doc, meta in retrieved:
        source = (meta or {}).get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{doc}")
    return "\n\n---\n\n".join(blocks)
