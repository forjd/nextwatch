-- Title embeddings. See docs/plans/04-content-similarity.md.
-- One row per title and model; the vibe "feel" sentences (plan 05) use their own model name.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE title_embedding (
    title_id     bigint      NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    model        text        NOT NULL,
    embedding    halfvec(384) NOT NULL,
    -- Hash of the text that was embedded; re-embed only when it changes.
    source_hash  text        NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (title_id, model)
);

-- Queries must repeat this WHERE clause to use the index.
CREATE INDEX title_embedding_bge_small_idx ON title_embedding
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE model = 'bge-small-en-v1.5';
