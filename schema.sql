-- Enable HNSW for fast vector search
create extension if not exists vector;

-- Nodes: Scientific Papers
create table if not exists papers (
  id uuid primary key default gen_random_uuid(),
  arxiv_id text unique not null,
  title text,
  abstract text,
  embedding vector(768), -- Fine-tuned SPECTER vectors
  metadata jsonb         -- Authors, Year, Journal
);

-- Index for fast cosine similarity search
create index if not exists papers_embedding_hnsw_idx 
on papers using hnsw (embedding vector_cosine_ops);

-- Edges: Explicit Citations (The Skeleton)
create table if not exists citations (
  source_id uuid references papers(id),
  target_id uuid references papers(id),
  primary key (source_id, target_id)
);

-- RPC Function: The "Implicit Matchmaker" (The Fascia)
-- Finds papers with structurally similar vectors (> 0.85 cosine similarity)
create or replace function match_papers(
  query_embedding vector(768), 
  match_threshold float, 
  match_count int
) returns table (id uuid, similarity float) language plpgsql as $$
begin
  return query
  select p.id, 1 - (p.embedding <=> query_embedding) as similarity
  from papers p
  where 1 - (p.embedding <=> query_embedding) > match_threshold
  order by similarity desc limit match_count;
end; $$;
