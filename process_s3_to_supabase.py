import json
import os
import time
from s3_manager import S3Manager
from data_engine import DataEngine

def process_s3_to_supabase():
    # Load env
    from verify_setup import load_env
    load_env()

    s3 = S3Manager()
    engine = DataEngine()

    # List raw files
    raw_files = s3.list_files(prefix="raw/")
    print(f"Found {len(raw_files)} files in raw/.")
    
    for key in raw_files:
        if not key.endswith(".jsonl"):
            continue
            
        print(f"Processing {key}...")
        local_filename = f"temp_{os.path.basename(key)}"
        
        # Download
        if not s3.download_file(key, local_filename):
            continue
            
        # Read and Parse
        papers = []
        with open(local_filename, 'r') as f:
            for line in f:
                if line.strip():
                    papers.append(json.loads(line))
        
        # Ingest to Supabase
        print(f"Vectorizing {len(papers)} papers...")
        try:
            engine.ingest_arxiv_batch(papers)
            print("Supabase Ingestion Successful.")
            
            # Move to processed
            processed_key = key.replace("raw/", "processed/")
            s3.move_file(key, processed_key)
            
        except Exception as e:
            print(f"Ingestion Failed: {e}")
        
        # Cleanup
        if os.path.exists(local_filename):
            os.remove(local_filename)

if __name__ == "__main__":
    process_s3_to_supabase()
