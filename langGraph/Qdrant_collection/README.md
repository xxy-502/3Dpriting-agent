# 3D Printing Hybrid RAG Qdrant Collection

This folder is the generated, offline-searchable database derived from
`../chunks/all_chunks.jsonl`.

## Current collection

- Collection name: `3d_printing_hybrid_rag_v1`
- Points: 955
- Dense retrieval: `BAAI/bge-small-en-v1.5`, 384 dimensions, cosine distance
- Sparse retrieval: corpus BM25, 9,719 terms
- Fusion: reciprocal rank fusion (RRF)
- Validation: 4/4 representative Hybrid queries passed

## Directory contents

- `qdrant_storage/`: persistent local Qdrant database. Do not edit its files
  manually and do not open it concurrently from multiple processes.
- `../../models/bge/`: local dense embedding model required for offline queries.
- `sparse/`: persisted BM25 vocabulary and weights needed to encode queries.
- `collection_config.json`: collection, vector, model, and fusion configuration.
- `_reports/`: build statistics and detailed Hybrid retrieval smoke tests.

The human-readable source of truth remains `../chunks/all_chunks.jsonl`.
The database can be rebuilt from that file and the local model.

## Query example

Run from the repository root:

```powershell
conda run -n pytorch python "3D Printing Knowledge Base\code\qdrant_collection\hybrid_search.py" "PETG stringing causes and retraction settings" --device cuda
```

See `../code/qdrant_collection/README.md` for build and validation commands.
