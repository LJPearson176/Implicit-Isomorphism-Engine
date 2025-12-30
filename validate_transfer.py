import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from econophysics import NeuralSDE

def load_and_prep(csv_path, cols):
    df = pd.read_csv(csv_path)
    data = df[cols].values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return torch.tensor(data, dtype=torch.float32)

def validate_transfer():
    print("--- Phase 8: Transfer Learning Validation ---")
    
    # 1. Load Physics Data (Source)
    # u, v => analogous to Drift, Diffusion or Price, Volatility
    try:
        source_data = load_and_prep("turbulence_ground_truth.csv", ['u', 'v'])
        print(f"Loaded Physics Data: {source_data.shape}")
    except FileNotFoundError:
        print("Error: turbulence_ground_truth.csv not found. Run physics_engine.py first.")
        return

    # 2. Load Finance Data (Target)
    # pressure (Price), temperature (Volatility)
    try:
        target_data = load_and_prep("lob_hydro_state.csv", ['pressure', 'temperature'])
        print(f"Loaded Finance Data: {target_data.shape}")
    except FileNotFoundError:
        print("Error: lob_hydro_state.csv not found. Run lob_engine.py first.")
        return

    # 3. Pre-train on Physics (The "Dual Lab")
    model = NeuralSDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nPre-training on Turbulence (Source)...")
    # Fake time input for simple prototype
    t_source = torch.linspace(0, 1, len(source_data)).view(-1, 1)
    
    for epoch in range(50):
        optimizer.zero_grad()
        # Input: State (u, v). Target: Next State (simplified)
        # In reality, we'd predict Drift/Diff.
        # Here, let's just train as an Autoencoder-like or Next-Step predictor to learn dynamics.
        
        drift, diff = model(torch.cat([t_source, source_data[:, 0:1]], dim=1)) # P=u
        
        # Loss: Reconstruction of 'v' (volatility/turbulence) from 'u'
        loss = criterion(diff, source_data[:, 1:2])
        loss.backward()
        optimizer.step()
    
    print(f"Pre-training Loss: {loss.item():.5f}")
    
    # 4. Transfer to Finance (Target)
    print("\nTransferring to Market LOB (Target)...")
    
    # Measure Zero-Shot Loss (Before Fine-tuning)
    t_target = torch.linspace(0, 1, len(target_data)).view(-1, 1)
    with torch.no_grad():
        drift, diff = model(torch.cat([t_target, target_data[:, 0:1]], dim=1))
        zero_shot_loss = criterion(diff, target_data[:, 1:2])
        
    print(f"Zero-Shot Transfer Loss: {zero_shot_loss.item():.5f}")
    
    # Fine-tune
    print("Fine-tuning...")
    for epoch in range(20):
        optimizer.zero_grad()
        drift, diff = model(torch.cat([t_target, target_data[:, 0:1]], dim=1))
        loss = criterion(diff, target_data[:, 1:2])
        loss.backward()
        optimizer.step()
        
    print(f"Fine-tuned Loss: {loss.item():.5f}")
    
    improvement = zero_shot_loss.item() - loss.item()
    print(f"\nTransfer Benefit (Adaptation): {improvement:.5f}")
    
    if loss.item() < 0.1:
         print("SUCCESS: Model transferred physics dynamics to market data effectively.")
    else:
         print("WARNING: Transfer loss remains high.")

if __name__ == "__main__":
    validate_transfer()
