# Knowledge base

Drop `.md`, `.txt` or `.pdf` files in this folder, then build the index:

```bash
python -m rag.ingest           # incremental
python -m rag.ingest --reset   # wipe and rebuild
```

Everything here is chunked (~800 chars, 100 overlap), embedded with
`text-embedding-3-small`, and stored in `../data/chroma_db` under the `voice_kb`
collection. The agent retrieves the top 4 chunks per question and is instructed
to say it does not know rather than invent an answer when they do not cover it.

Leave the folder empty and the agent still works — `retrieve()` short-circuits on
an empty collection, so you do not pay the embedding round trip for nothing.

This file is itself indexable, so a fresh clone has something to answer questions
about. Ask the running agent "how do I add documents to the knowledge base?" to
confirm retrieval is working end to end.
