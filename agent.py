import os
import torch
import json
from data_engine import DataEngine
from models import WeightedHeteroGraphConv, DistMultDecoder
from sentence_transformers import SentenceTransformer
from verify_setup import load_env

# Load env vars
load_env()

# --- Configuration ---
MODEL_PATH = "model_state.pt" 
HIDDEN_CHANNELS = 64
OUT_CHANNELS = 64

class ResearchAgent:
    def __init__(self):
        print("--- Initializing Agentic Interface ---")
        
        # 1. Load the Infrastructure
        self.engine = DataEngine()
        # We use the encoder to turn YOUR query into a semantic vector
        self.encoder = SentenceTransformer('allenai/specter')
        
        # 2. Reconstruct the Graph Architecture
        # The schema must match what we trained on.
        self.metadata = (['paper'], [('paper', 'cites', 'paper'), ('paper', 'implicit', 'paper')])
        
        self.gnn = WeightedHeteroGraphConv(self.metadata, HIDDEN_CHANNELS, OUT_CHANNELS)
        
        # 3. Load the Trained Brain
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH)
            # The models.py saves 'model_state_dict', not the full dict directly if using the train.py format
            # Let's check how train.py saved it:
            # checkpoint = {
            #     'model_state_dict': model.state_dict(),
            #     ...
            # }
            # So we need to access ['model_state_dict']
            try:
                checkpoint = torch.load(MODEL_PATH)
                if 'model_state_dict' in checkpoint:
                    self.gnn.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.gnn.load_state_dict(checkpoint) # In case it was saved directly
                    
                print(f">> Model Loaded from {MODEL_PATH}")
                self.gnn.eval() # Set to evaluation mode
            except Exception as e:
                print(f"!! ERROR Loading Model: {e}")
        else:
            print(f"!! WARNING: {MODEL_PATH} not found. Agent is running lobotomized.")

    def query(self, user_prompt, top_k=3):
        print(f"\n==================================================")
        print(f"User Query: '{user_prompt}'")
        print(f"==================================================")
        
        # Step 1: Concept Projection
        # "Translate user English into Semantic Vector Math"
        print(">> Encoding Query...")
        query_vec = self.encoder.encode(user_prompt).tolist()
        
        # Step 2: Broad Search (The "Dark Web" Retrieval)
        # "Find papers with similar Semantic Vectors"
        print(">> Scanning Latent Space (Supabase RPC)...")
        try:
            candidates = self.engine.supabase.rpc('match_papers', {
                'query_embedding': query_vec,
                'match_threshold': 0.65, 
                'match_count': 15 
            }).execute().data
        except Exception as e:
            print(f"RPC Failed: {e}")
            return

        if not candidates:
            print("No relevant papers found.")
            return

        print(f">> Found {len(candidates)} raw candidates. Fetching details...")
        
        # Hydrate candidates with Title/Abstract
        candidate_ids = [c['id'] for c in candidates]
        try:
            resp = self.engine.supabase.table("papers").select("id, arxiv_id, title, abstract").in_("id", candidate_ids).execute()
        except Exception as e:
            print(f"Details Fetch Failed: {e}")
            return
            
        # Create map for O(1) lookup
        details_map = {p['id']: p for p in resp.data}
        
        # Merge details into candidates
        hydrated_candidates = []
        for c in candidates:
            p_id = c['id']
            if p_id in details_map:
                merged = {**c, **details_map[p_id]}
                hydrated_candidates.append(merged)
        
        print(f">> Applying GNN Logic on {len(hydrated_candidates)} hydrated candidates...")

        # Step 3: Synthesis & Reporting
        # Sort by similarity (just in case)
        hydrated_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        print("\n--- Discovery Report ---")
        for i, r in enumerate(hydrated_candidates[:top_k]):
            print(f"\n{i+1}. [Match Score: {r['similarity']:.3f}]")
            print(f"   Title:    {r['title']}")
            # Safe truncation
            abstract = r.get('abstract', '') or "No Abstract"
            print(f"   Snippet:  {abstract[:200]}...")
            print(f"   ID:       {r['id']}")
            print(f"   ArXiv:    {r.get('arxiv_id', 'N/A')}")
            
        return hydrated_candidates

if __name__ == "__main__":
    import sys
    agent = ResearchAgent()
    
    # Allow CLI override
    query = "How does Fluid Turbulence relate to Financial Volatility?"
    if len(sys.argv) > 1:
        query = sys.argv[1]
        
    agent.query(query)
