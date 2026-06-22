# Knowledge-base retrieval scaling (#I3)

`DocumentRetriever.retrieve()` scores a query against every row in
`knowledge_documents` with a **numpy cosine full scan** (O(N)). This runs on every
teacher correction in every curation cycle, so latency scales linearly with KB
size. At hundreds–low-thousands of docs this is fine; past a few thousand it
becomes a bottleneck.

## Guardrail (implemented)

`POST /knowledge/documents` returns a `warning` and increments
`knowledge_base_size_warnings_total` once the KB exceeds
`KNOWLEDGE_BASE_SIZE_WARN_THRESHOLD` (default 5000). The `knowledge_base_documents`
gauge tracks current size for dashboards/alerts.

## Migration path: pgvector + IVFFlat (sub-linear ANN)

PostgreSQL 15+ supports the `pgvector` extension. When the KB grows large:

1. `CREATE EXTENSION IF NOT EXISTS vector;`
2. Add a `vector(384)` column (MiniLM dim) alongside the existing JSONB embedding,
   backfill it, then build an ANN index:
   ```sql
   ALTER TABLE knowledge_documents ADD COLUMN embedding_vec vector(384);
   -- backfill embedding_vec from embedding JSONB ...
   CREATE INDEX ON knowledge_documents USING ivfflat (embedding_vec vector_cosine_ops)
     WITH (lists = 100);
   ```
3. Switch `retrieve()` to `ORDER BY embedding_vec <=> :query LIMIT k` (cosine
   distance), keeping the numpy scan as the fallback when the extension is absent
   (detect via `SELECT 1 FROM pg_extension WHERE extname='vector'`).

This keeps the design's "no hard pgvector dependency" property while giving a
clean, additive route to ANN search when scale demands it.
