import os
import torch
from sentence_transformers import SentenceTransformer
import supabase
from supabase import create_client, Client
from torch_geometric.data import HeteroData

class DataEngine:
    def __init__(self):
        # Initialize Supabase Client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        self.supabase: Client = create_client(url, key)

        # Initialize Embedding Model
        # Check for fine-tuned model first, else fallback to base
        fine_tuned_path = "./models/specter_matchmaker_v1"
        if os.path.exists(fine_tuned_path):
            print(f"Loading fine-tuned model from {fine_tuned_path}")
            self.model = SentenceTransformer(fine_tuned_path)
        else:
            print("Loading base SPECTER model")
            self.model = SentenceTransformer('allenai/specter')

    def ingest_arxiv_batch(self, papers):
        """
        Ingest a batch of papers into Supabase.
        papers: List of dicts with keys: arxiv_id, title, abstract, metadata
        """
        # Prepare inputs for embedding
        # SPECTER format: title + [SEP] + abstract
        texts = [p['title'] + ' [SEP] ' + p['abstract'] for p in papers]
        embeddings = self.model.encode(texts)

        data = []
        for i, paper in enumerate(papers):
            record = {
                "arxiv_id": paper['arxiv_id'],
                "title": paper['title'],
                "abstract": paper['abstract'],
                "metadata": paper.get('metadata', {}),
                "embedding": embeddings[i].tolist()
            }
            data.append(record)

        # Upsert to Supabase
        # Chunk upserts to avoid timeout
        batch_size = 50
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            try:
                self.supabase.table("papers").upsert(chunk, on_conflict="arxiv_id").execute()
                print(f"Upserted batch {i} to {i+len(chunk)}")
            except Exception as e:
                print(f"Error upserting chunk {i}: {e}")
                # Optional: break or continue? Let's re-raise to signal failure for the file
                raise e
                
        return True # success indicator instead of raw response

    def build_graph_batch(self, paper_ids, match_threshold=0.85):
        """
        Build a HeteroData object for a batch of papers (Subgraph).
        this fetches explicit citations AND implicit matches.
        """
        # 1. Fetch Paper Nodes (features) from Supabase
        # We need to query by IDs. 
        # For simplicity in this prototype, let's assume paper_ids are UUIDs present in DB.
        
        # In a real scenario, we might want to fetch the exact rows for paper_ids.
        # But here, let's assume we proceed to find neighbors for these papers 
        # to build a valid subgraph.
        
        # For this prototype, let's just create a graph *among* the provided paper_ids
        # OR extend to 1-hop neighbors? 
        # The prompt says: "Dynamic Graph Building: For every batch, call match_papers..."
        
        # Implementation strategy:
        # Iterate over paper_ids, get their embeddings (fetched or re-computed?), 
        # finding implicit edges to OTHER discovery candidates.
        
        # Fetch papers in chunks to avoid URL length / payload limits
        chunk_size = 200
        papers_data = []
        
        for i in range(0, len(paper_ids), chunk_size):
            chunk_ids = paper_ids[i : i + chunk_size]
            try:
                resp = self.supabase.table("papers").select("*").in_("id", chunk_ids).execute()
                papers_data.extend(resp.data)
            except Exception as e:
                print(f"Error fetching chunk {i}: {e}")
                
        if not papers_data:
            return HeteroData()

        # Create mapping from UUID to index
        id_to_idx = {p['id']: i for i, p in enumerate(papers_data)}
        
        # 2. Build Node Features
        # x: [num_papers, embedding_dim]
        # Check if embedding is string (supabase/pgvector quirk) and parse
        import json
        embeddings = []
        for p in papers_data:
            emb = p['embedding']
            if isinstance(emb, str):
                # Simple parsing for string representation of vector
                # pgvector likely returns "[0.1, 0.2, ...]"
                emb = json.loads(emb)
            embeddings.append(emb)
            
        x = torch.tensor(embeddings, dtype=torch.float)
        
        data = HeteroData()
        data['paper'].x = x
        data['paper'].num_nodes = len(papers_data)
        data['paper'].paper_ids = [p['id'] for p in papers_data] # Store UUIDs for reference

        # 3. Build Edges
        explicit_edges = [[], []]
        implicit_edges = [[], []]

        # Optimization: Batch processing instead of N+1 queries
        paper_uuids = [p['id'] for p in papers_data]

        # 1. Implicit Matches (Batch RPC)
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rpc_params = {
                    "query_ids": paper_uuids,
                    "match_threshold": match_threshold,
                    "match_count": 10
                }
                # Note: Requires 'match_papers_batch' RPC to be installed in DB
                matches = self.supabase.rpc("match_papers_batch", rpc_params).execute()
                
                for m in matches.data:
                    source_id = m['source_id']
                    target_id = m['target_id']

                    if source_id in id_to_idx and target_id in id_to_idx:
                        src_idx = id_to_idx[source_id]
                        target_idx = id_to_idx[target_id]

                        if src_idx != target_idx:
                            implicit_edges[0].append(src_idx)
                            implicit_edges[1].append(target_idx)
                break # Success
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Batch RPC Error: {e}")
                time.sleep(0.5 * (attempt + 1))

        # 2. Explicit Citations (Batch Query)
        for attempt in range(max_retries):
            try:
                # Fetch all citations where source is in our batch
                citations = self.supabase.table("citations") \
                    .select("source_id, target_id") \
                    .in_("source_id", paper_uuids) \
                    .execute()

                for c in citations.data:
                    source_id = c['source_id']
                    target_id = c['target_id']

                    if source_id in id_to_idx and target_id in id_to_idx:
                        src_idx = id_to_idx[source_id]
                        target_idx = id_to_idx[target_id]
                        explicit_edges[0].append(src_idx)
                        explicit_edges[1].append(target_idx)
                break # Success
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Batch Citation Error: {e}")
                time.sleep(0.5 * (attempt + 1))

        if explicit_edges[0]:
            data['paper', 'cites', 'paper'].edge_index = torch.tensor(explicit_edges, dtype=torch.long)
        else:
            data['paper', 'cites', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        if implicit_edges[0]:
            data['paper', 'implicit', 'paper'].edge_index = torch.tensor(implicit_edges, dtype=torch.long)
        else:
            data['paper', 'implicit', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data
