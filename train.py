import torch
import torch.nn as nn
import torch.optim as optim
from data_engine import DataEngine
from models import WeightedHeteroGraphConv, DistMultDecoder
from torch_geometric.data import HeteroData
from verify_setup import load_env

# Load environment variables first
load_env()

def train_matchmaker(epochs=100, lr=0.001):
    print("Initializing Data Engine...")
    try:
        engine = DataEngine()
    except Exception as e:
        print(f"Data Engine Init Failed (Expected if Env Vars missing): {e}")
        return

    # 1. Load Data (Prototype: Mock Batch)
    # In production, we'd fetch specific papers or walk the graph.
    # For now, let's assume we have a list of paper IDs to train on.
    # This part would need actual IDs from the DB.
    # For the sake of the script being runnable in a real env, we will try to fetch some random papers 
    # or just rely on the user to provide IDs. 
    # Let's mock the data object construction if we can't connect, 
    # but the instruction implies a functional script.
    
    # Let's assume the user will run this with valid credentials.
    # We will try to fetch a small batch if possible, or fail gracefully.
    print("Fetching data batch (Full Dataset)...")
    print("Fetching data batch (Full Dataset)...")
    # Fetch all IDs with pagination
    paper_ids = []
    offset = 0
    limit = 1000
    while True:
        try:
            resp = engine.supabase.table("papers").select("id").range(offset, offset + limit - 1).execute()
            batch_ids = [row['id'] for row in resp.data]
            if not batch_ids:
                break
            paper_ids.extend(batch_ids)
            offset += limit
            print(f"Fetched {len(paper_ids)} IDs...")
            if len(batch_ids) < limit:
                break
        except Exception as e:
            print(f"Error fetching IDs: {e}")
            break
            
    if not paper_ids:
        print("No papers found in DB. Please ingest data first.")
        return
    
    data = engine.build_graph_batch(paper_ids)

    print(f"Graph Built: {data}")
    
    # 2. Initialize Models
    # Metadata: (node_types, edge_types)
    metadata = data.metadata()
    hidden_channels = 64
    out_channels = 64
    
    # Encoder
    model = WeightedHeteroGraphConv(metadata, hidden_channels=hidden_channels, out_channels=out_channels)
    
    # Decoder
    decoder = DistMultDecoder(metadata, hidden_channels=out_channels)
    
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=lr)

    # 3. Training Loop
    model.train()
    decoder.train()
    
    # Check for NaNs/Infs in input features
    if torch.isnan(data.x_dict['paper']).any() or torch.isinf(data.x_dict['paper']).any():
        print("CRITICAL ERROR: Input node features contain NaNs or Infs!")
        return

    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward Pass (Encoder)
        
        # --- PHASE 25 SELF-EVOLUTION HOOK ---
        # If the system has generated its own optimizer, use it.
        try:
            from generated_optimizer import self_optimize_graph
            # Apply Synaptic Pruning
            data.edge_index_dict = self_optimize_graph(data.edge_index_dict, z_dict if 'z_dict' in locals() else None, epoch)
        except ImportError:
            pass # No evolution yet
        # ------------------------------------

        z_dict = model(data.x_dict, data.edge_index_dict)
        
        # Check embeddings for NaNs
        if torch.isnan(z_dict['paper']).any():
             print(f"Epoch {epoch}: NaN detected in node embeddings (z_dict)")
             break
             
        # Forward Pass (Decoder) -> Link Prediction
        
        # 1. Positive Scores
        pos_scores_dict = decoder(z_dict, data.edge_index_dict)
        
        if not pos_scores_dict:
            continue
            
        # 2. Negative Scores
        neg_edge_index_dict = {}
        for edge_type, edge_index in data.edge_index_dict.items():
            src_t, rel, dst_t = edge_type
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue
                
            num_dst_nodes = data[dst_t].num_nodes
            rand_dst = torch.randint(0, num_dst_nodes, (num_edges,), device=edge_index.device)
            neg_edge_index = torch.stack([edge_index[0], rand_dst], dim=0)
            neg_edge_index_dict[edge_type] = neg_edge_index

        neg_scores_dict = decoder(z_dict, neg_edge_index_dict)

        # 3. Margin Ranking Loss
        margin = 1.0
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        for edge_type, pos_score in pos_scores_dict.items():
            if edge_type in neg_scores_dict:
                neg_score = neg_scores_dict[edge_type]
                
                # DEBUG: Score Stats
                if epoch == 0:
                    print(f"DEBUG: Edge {edge_type} | Pos Mean: {pos_score.mean().item():.3f} | Neg Mean: {neg_score.mean().item():.3f}")
                
                # Hinge loss
                loss = torch.clamp(margin - pos_score + neg_score, min=0).mean()
                total_loss = total_loss + loss
        
        if total_loss.item() != 0.0:
            total_loss.backward()
            
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {total_loss.item():.4f}")

    # 4. Audit Step
    print("\n--- AUDIT: Learned Trust Weights ---")
    for name, param in model.edge_trust_weights.items():
        # param is the alpha. Trust score is sigmoid(alpha).
        trust_score = torch.sigmoid(param).item()
        print(f"Edge Type: {name} | Raw Alpha: {param.item():.4f} | Trust Score: {trust_score:.4f}")

    print("------------------------------------")
    print("Interpretation:")
    print("If 'implicit' weights are high (>0.5), the model is valuing methodological links!")

    # Save Model
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs
    }
    torch.save(checkpoint, "model_state.pt")
    print("Model saved to model_state.pt")

if __name__ == "__main__":
    train_matchmaker()
