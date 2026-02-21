-- Batch version of match_papers to avoid N+1 queries
create or replace function match_papers_batch(
  query_ids uuid[],
  match_threshold float,
  match_count int
) returns table (source_id uuid, target_id uuid, similarity float) language plpgsql as $$
begin
  return query
  select q.id as source_id, m.id as target_id, m.similarity
  from papers q
  cross join lateral (
    select p.id, 1 - (p.embedding <=> q.embedding) as similarity
    from papers p
    where 1 - (p.embedding <=> q.embedding) > match_threshold
    order by similarity desc limit match_count
  ) m
  where q.id = any(query_ids);
end; $$;
