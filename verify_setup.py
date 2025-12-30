import os
import sys

# manually load .env since we might not have python-dotenv
def load_env():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
                    print(f"Loaded {key}")

load_env()

try:
    from data_engine import DataEngine
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def verify():
    print("Initializing DataEngine...")
    engine = DataEngine()
    
    # 1. Ingest a mock paper
    mock_paper = {
        "arxiv_id": "test_001",
        "title": "Liquid Time-Constant Networks",
        "abstract": "Continuous-time neural networks with adaptive time constants.",
        "metadata": {"year": 2021, "authors": ["Hasani et al."]}
    }
    
    print(f"Ingesting mock paper: {mock_paper['title']}...")
    try:
        engine.ingest_arxiv_batch([mock_paper])
        print("Ingestion successful.")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
        return

    # 2. Test RPC match_papers
    # We query with the same paper's embedding (simulated) or just check connectivity
    # Re-embed to get the vector
    vector = engine.model.encode(mock_paper['title'] + ' [SEP] ' + mock_paper['abstract']).tolist()
    
    print("Testing RPC 'match_papers'...")
    try:
        rpc_result = engine.supabase.rpc("match_papers", {
            "query_embedding": vector,
            "match_threshold": 0.5, # Lower threshold to ensure we match ourself
            "match_count": 5
        }).execute()
        
        print(f"RPC Result: Found {len(rpc_result.data)} matches.")
        for match in rpc_result.data:
            print(f" - ID: {match['id']}, Similarity: {match['similarity']}")
            
    except Exception as e:
        print(f"RPC Failed: {e}")
        return
        
    print("\nVERIFICATION COMPLETE: Supabase connection, Ingestion, and RPC are working.")

if __name__ == "__main__":
    verify()
