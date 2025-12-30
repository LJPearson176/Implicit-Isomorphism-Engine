import arxiv
import json
import time
import os
from s3_manager import S3Manager
from supabase import create_client

def ingest_arxiv_to_s3(query="cat:cs.AI", max_results=300):
    # Initialize Supabase for Deduplication Check
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    supabase = None
    if url and key:
        supabase = create_client(url, key)

    print(f"Searching ArXiv for: {query} (Limit: {max_results})...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = []
    fetched_ids = []
    
    # 1. Fetch from ArXiv
    raw_papers = []
    for r in client.results(search):
        paper_id = r.entry_id.split('/')[-1]
        raw_papers.append({
            "arxiv_id": paper_id,
            "title": r.title,
            "abstract": r.summary,
            "metadata": {
                "authors": [a.name for a in r.authors],
                "published": str(r.published),
                "journal_ref": r.journal_ref
            }
        })
        fetched_ids.append(paper_id)
    
    print(f"Fetched {len(raw_papers)} papers from ArXiv.")
    
    # 2. Check Duplicates in Supabase
    if supabase and fetched_ids:
        try:
            # Check which IDs already exist
            # Supabase 'in_' filter limit is usually high enough for batch sizes < 1000
            existing_rows = supabase.table("papers").select("arxiv_id").in_("arxiv_id", fetched_ids).execute()
            existing_ids = {row['arxiv_id'] for row in existing_rows.data}
            
            # Filter
            new_papers = [p for p in raw_papers if p['arxiv_id'] not in existing_ids]
            print(f"Deduplication: {len(existing_ids)} already exist. {len(new_papers)} are new.")
            results = new_papers
        except Exception as e:
            print(f"Warning: Deduplication check failed ({e}). Proceeding with all.")
            results = raw_papers
    else:
        results = raw_papers

    if not results:
        print("No new papers to upload.")
        return

    # 3. Save to local JSONL
    filename = f"arxiv_batch_{int(time.time())}.jsonl"
    with open(filename, 'w') as f:
        for p in results:
            f.write(json.dumps(p) + "\n")
            
    # 4. Upload to S3
    s3 = S3Manager()
    s3_key = f"raw/{filename}"
    success = s3.upload_file(filename, s3_key)
    
    if success:
        os.remove(filename)
        print("Local file cleaned up.")
    else:
        print("Upload failed, local file preserved.")

def ingest_strategic_corpus():
    # Configuration based on research_strategy.md
    SEARCH_CONFIGS = [
        # --- PHYSICS DOMAIN (Source) ---
        {
            "query": "cat:cond-mat.stat-mech", # Statistical Mechanics (Ising -> Agents)
            "max_results": 1500
        },
        {
            "query": "cat:physics.flu-dyn", # Fluid Dynamics (Navier-Stokes)
            "max_results": 1500
        },
        {
            "query": "cat:physics.soc-ph", # Econophysics (Society)
            "max_results": 1000
        },
        
        # --- FINANCE DOMAIN (Target) ---
        {
            "query": "cat:q-fin.ST", # Statistical Finance (Fat Tails)
            "max_results": 1500
        },
        {
            "query": "cat:q-fin.TR", # Trading & Microstructure (LOB)
            "max_results": 1500
        },
        {
            "query": "cat:q-fin.MF", # Mathematical Finance (Pricing)
            "max_results": 1000
        },
        
        # --- BASELINE (AI/Bio) ---
        { "query": "cat:cs.LG", "max_results": 500 },
        { "query": "cat:q-bio.NC", "max_results": 500 },
        
        # --- PHASE 14: ECOLOGY DOMAIN ---
        { "query": "cat:q-bio.PE", "max_results": 500 }, # Populations and Evolution

        # --- PHASE 16: SOCIAL PHYSICS DOMAIN ---
        { "query": "cat:cs.SI", "max_results": 500 }, # Social and Information Networks

        # --- PHASE 17: QUANTUM DOMAIN ---
        { "query": "cat:quant-ph", "max_results": 500 }, # Quantum Physics

        # --- PHASE 31: CYBERNETICS DOMAIN (Control Theory) ---
        { "query": "cat:eess.SY", "max_results": 500 }   # Systems and Control
    ]

    print(f"--- Starting Strategic Corpus Ingestion (Target: ~9000 papers) ---")
    for config in SEARCH_CONFIGS:
        print(f"\n>> Ingesting Silo: {config['query']}")
        ingest_arxiv_to_s3(
            query=config['query'], 
            max_results=config['max_results']
        )

if __name__ == "__main__":
    # Load env (if not using dotenv via cli)
    from verify_setup import load_env
    load_env()
    
    # Run the Strategic Ingestion
    ingest_strategic_corpus()
