-- LLM vibe tags. See docs/plans/05-vibe-tags.md.

CREATE TABLE title_vibe (
    title_id        bigint      NOT NULL REFERENCES title (id) ON DELETE CASCADE,
    model           text        NOT NULL,
    prompt_version  integer     NOT NULL,
    tags            jsonb       NOT NULL,
    feel            text        NOT NULL,
    input_tokens    integer,
    output_tokens   integer,
    cost_usd        numeric(10, 6),
    created_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (title_id, model)
);

CREATE INDEX title_vibe_created_idx ON title_vibe (created_at);

-- Embeddings of the one-sentence "feel", for mood search.
CREATE INDEX title_embedding_feel_idx ON title_embedding
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE model = 'bge-small-en-v1.5:feel';
