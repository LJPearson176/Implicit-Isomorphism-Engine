import os
import json
import torch
import arxiv
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from data_engine import DataEngine
from verify_setup import load_env

def visualize_bridge():
    load_env()
    print("--- Initializing Bridge Visualization ---")
    engine = DataEngine()
    
    # 1. Fetch Papers
    print("Fetching all papers...")
    # Fetch IDs and Embeddings (and metadata if any)
    # Pagination might be needed if > 1000, simplified loop here
    papers = []
    offset = 0
    limit = 1000
    while True:
        resp = engine.supabase.table("papers").select("id, arxiv_id, title, embedding, metadata").range(offset, offset + limit - 1).execute()
        batch = resp.data
        if not batch:
            break
        papers.extend(batch)
        offset += limit
        print(f"Fetched {len(papers)} papers...")
        if len(batch) < limit:
            break

    if not papers:
        print("No papers found.")
        return

    # 2. Hydrate Categories (if missing)
    print("Hydrating Categories from ArXiv...")
    # Filter invalid IDs (e.g. test_001)
    arxiv_ids = [p['arxiv_id'] for p in papers]
    valid_ids = [aid for aid in arxiv_ids if aid and 'test_' not in aid and '.' in aid]
    # Random Sample if too large
    import random
    if len(valid_ids) > 600:
        print(f"Dataset too large ({len(valid_ids)}). Sampling 600 random papers for visualization...")
        valid_ids = random.sample(valid_ids, 600)
    
    print(f"Optimizing fetch for {len(valid_ids)} papers...")
    
    # Process in chunks of 100 for ArXiv API (Faster)
    id_to_category = {}
    chunk_size = 100
    client = arxiv.Client(
        page_size=chunk_size,
        delay_seconds=3.0, 
        num_retries=3
    )
    
    for i in range(0, len(valid_ids), chunk_size):
        chunk = valid_ids[i:i+chunk_size]
        search = arxiv.Search(id_list=chunk)
        try:
            print(f"Fetching chunk {i // chunk_size + 1}...")
            for r in client.results(search):
                p_id = r.entry_id.split('/')[-1]
                # Store primary category
                id_to_category[p_id] = r.primary_category
        except Exception as e:
            print(f"ArXiv Fetch Error on chunk {i}: {e}")
            
    print(f"Resolved categories for {len(id_to_category)} papers.")
    
    # 3. Build Graph
    print("Computing Similarity Graph...")
    embeddings = []
    valid_papers = []
    colors = []
    
    for p in papers:
        aid = p['arxiv_id']
        cat = id_to_category.get(aid, 'unknown')
        
        # Determine Color/Silo
        # Determine Color/Silo
        if 'quant-ph' in cat:
            color = 'purple' # Quantum
            silo = 'Quantum'
        elif 'q-fin' in cat:
            color = 'green' # Finance
            silo = 'Finance'
        elif 'q-bio' in cat:
            color = 'red' # Biology/Ecology
            silo = 'Bio'
        elif 'cs.SI' in cat or 'soc-ph' in cat:
            color = 'orange' # Social Science (Yellow is hard to see)
            silo = 'Social'
        elif 'physics' in cat or 'cond-mat' in cat:
            color = 'blue' # Core Physics
            silo = 'Physics'
        elif 'cs.' in cat or 'stat.' in cat:
            color = 'grey' # AI/Methods
            silo = 'AI'
        else:
            color = 'black'
            silo = 'Other'
            
        # Parse embedding
        emb = p['embedding']
        if isinstance(emb, str):
            emb = json.loads(emb)
            
        embeddings.append(emb)
        valid_papers.append({'id': p['id'], 'arxiv_id': aid, 'silo': silo, 'color': color})
        colors.append(color)

    if not embeddings:
        return

    X = np.array(embeddings)
    # Cosine Similarity Matrix
    sim_matrix = cosine_similarity(X)
    
    # NetworkX
    G = nx.Graph()
    
    # Add Nodes
    for i, p in enumerate(valid_papers):
        G.add_node(i, silo=p['silo'], color=p['color'])
        
    # Add Edges (Threshold > 0.85)
    print("Adding Edges...")
    num_papers = len(valid_papers)
    bridge_edges = []
    internal_edges = []
    
    threshold = 0.85
    
    # Identify edges
    # Use indices to avoid double counting, upper triangle
    rows, cols = np.where(np.triu(sim_matrix, 1) > threshold)
    
    for r, c in zip(rows, cols):
        silo_a = valid_papers[r]['silo']
        silo_b = valid_papers[c]['silo']
        
        if silo_a != silo_b and silo_a != 'Other' and silo_b != 'Other':
            # This is a BRIDGE!
            bridge_edges.append((r, c))
        else:
            internal_edges.append((r, c))
            
    print(f"Found {len(internal_edges)} internal edges and {len(bridge_edges)} BRIDGE EDGES.")

    # Save Bridge Candidates for Compiler
    bridge_candidates = []
    for r, c in bridge_edges:
        src_p = valid_papers[r]
        dst_p = valid_papers[c]
        bridge_candidates.append({
            "source_id": src_p['id'],
            "source_arxiv": src_p['arxiv_id'],
            "source_silo": src_p['silo'],
            "target_id": dst_p['id'], 
            "target_arxiv": dst_p['arxiv_id'],
            "target_silo": dst_p['silo'],
            "similarity": float(sim_matrix[r][c])
        })
    
    with open("bridge_edges.json", "w") as f:
        json.dump(bridge_candidates, f, indent=2)
    print(f"Saved {len(bridge_candidates)} bridge candidates to bridge_edges.json")

    # 4. Plot
    print("Rendering Visualization...")
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=0.15, iterations=50) # Force-directed layout
    
    # Draw Nodes
    node_colors = [G.nodes[i]['color'] for i in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color=node_colors, alpha=0.7)
    
    # Draw Internal Edges (Faint)
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, alpha=0.05, edge_color='grey')
    
    # Draw Bridge Edges (Highlighted)
    if bridge_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bridge_edges, alpha=0.8, edge_color='red', width=1.5)
        
    plt.title(f"Grand Unified Map: Physics (Blue), Finance (Green), Bio (Red), Social (Orange), Quantum (Purple)\nRed Lines = Implicit Isomorphisms (Sim > {threshold})")
    plt.axis('off')
    
    # Save
    output_path = "bridge_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_bridge()
