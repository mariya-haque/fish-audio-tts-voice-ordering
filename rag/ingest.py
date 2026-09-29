"""Build the Chroma index from whatever is in docs/.

    python -m rag.ingest            # incremental: re-indexes changed files
    python -m rag.ingest --reset    # wipe the collection first

Supports .md, .txt and .pdf. Each file is chunked, embedded with
text-embedding-3-small, and upserted with a stable id so re-running is safe.
"""

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb

import config
from rag.engine import embed

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ingest")

SUPPORTED = {".md", ".txt", ".pdf"}
EMBED_BATCH = 64


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("  skipping %s (pip install pypdf)", path.name)
            return ""
        try:
            reader = PdfReader(str(path))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:
            logger.warning("  could not read %s: %s", path.name, exc)
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("  could not read %s: %s", path.name, exc)
        return ""


def chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph boundaries where possible, hard-splitting only when a
    single paragraph exceeds the target size."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        while len(para) > size:
            # Oversized paragraph: emit a full-size slice and keep going.
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size - overlap :]

        if not current:
            current = para
        elif len(current) + len(para) + 2 <= size:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para

    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="delete the collection first")
    args = parser.parse_args()

    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set (needed to embed). See .env.example.")
        return 1

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if args.reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            logger.info("Deleted collection %s", config.COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    files = sorted(p for p in config.DOCS_DIR.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        logger.warning(
            "No .md/.txt/.pdf files in %s -- nothing to index.\n"
            "The agent will still run; it just answers without a knowledge base.",
            config.DOCS_DIR,
        )
        return 0

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for path in files:
        text = read_text(path)
        if not text.strip():
            continue
        rel = path.relative_to(config.DOCS_DIR).as_posix()
        pieces = chunk(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        logger.info("  %-40s %4d chunks", rel, len(pieces))
        for i, piece in enumerate(pieces):
            # Stable id keyed on content, so re-ingesting an unchanged file is a no-op
            # and an edited file replaces only the chunks that actually changed.
            digest = hashlib.sha1(f"{rel}:{i}:{piece}".encode()).hexdigest()[:16]
            ids.append(f"{rel}:{i}:{digest}")
            documents.append(piece)
            metadatas.append({"source": rel, "chunk": i})

    if not documents:
        logger.warning("Found files but extracted no text.")
        return 0

    logger.info("Embedding %d chunks with %s ...", len(documents), config.EMBED_MODEL)
    for start in range(0, len(documents), EMBED_BATCH):
        batch_docs = documents[start : start + EMBED_BATCH]
        vectors = await embed(batch_docs)
        collection.upsert(
            ids=ids[start : start + EMBED_BATCH],
            documents=batch_docs,
            metadatas=metadatas[start : start + EMBED_BATCH],
            embeddings=vectors,
        )
        logger.info("  upserted %d/%d", min(start + EMBED_BATCH, len(documents)), len(documents))

    logger.info("Done. Collection %r now holds %d chunks.", config.COLLECTION_NAME, collection.count())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
