import torch
import torch.nn as nn
import numpy as np
import os
from verify_setup import load_env

class FinanceLoader:
    def __init__(self, n_points=1000, dt=0.01):
        self.n_points = n_points
        self.dt = dt
        
    def load_real_data(self, filepath="lob_hydro_state.csv"):
        """
        Loads real Hydrodynamic LOB data if available.
        Expected columns: [pressure, temperature, mass_flux, velocity] (Index: Time)
        """
        if not os.path.exists(filepath):
            print(f">> [FinanceLoader] Real data '{filepath}' not found. Falling back to synthetic.")
            return None, None
            
        print(f">> [FinanceLoader] Loading real LOB data from {filepath}...")
        import pandas as pd
        df = pd.read_csv(filepath, parse_dates=["T"])
        
        # Normalize Time to [0, T_max]
        t_raw = (df["T"] - df["T"].min()).dt.total_seconds().values
        t = torch.tensor(t_raw, dtype=torch.float32).view(-1, 1)
        
        # Pressure is Price
        P = torch.tensor(df["pressure"].values, dtype=torch.float32).view(-1, 1)
        
        # Update internal dt estimate
        if len(t) > 1:
            self.dt = (t[1] - t[0]).item()
            
        return t, P

    def generate_synthetic_data(self):
        """
        Generates Geometric Brownian Motion (GBM) as a proxy for stock prices.
        dS = mu*S*dt + sigma*S*dW
        """
        mu = 0.1
        sigma = 0.2
        S0 = 100.0
        
        t = np.linspace(0, self.n_points*self.dt, self.n_points)
        W = np.random.standard_normal(size=self.n_points)
        W = np.cumsum(W)*np.sqrt(self.dt) ### standard brownian motion ###
        X = (mu - 0.5*sigma**2)*t + sigma*W
        S = S0 * np.exp(X)
        
        return torch.tensor(t, dtype=torch.float32).view(-1, 1), torch.tensor(S, dtype=torch.float32).view(-1, 1)

class NeuralSDE(nn.Module):
    """
    Approximates the Drift (f) and Diffusion (g) of a stochastic process.
    dPh/dt = f(Ph, t) + g(Ph, t) * noise
    """
    def __init__(self):
        super().__init__()
        # Drift Network: f(P, t) -> Trend
        self.drift_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Diffusion Network: g(P, t) -> Volatility
        self.diff_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x is [Time, Price]
        drift = self.drift_net(x)
        diff = self.diff_net(x)
        return drift, diff

class PhysicsLoss(nn.Module):
    """
    Calculates the residual of the 'Financial Hydrodynamics' equation found in Phase 6.
    Here we enforce a simple Momentum Conservation analog:
    dP/dt + P * grad(P) = -grad(Liquidity) + Volatility * laplacian(P)
    
    Simplified for 1D time-series:
    dP/dt - Drift = 0 
    (In full PINN, we would calculate spatial gradients, but for 1D we check consistency)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
        # Dynamic Load
        self.custom_loss = None
        try:
            import generated_loss
            import importlib
            importlib.reload(generated_loss) # Force reload if changed
            if hasattr(generated_loss, 'custom_physics_loss'):
                self.custom_loss = generated_loss.custom_physics_loss
                print(">> [PhysicsLoss] Loaded AI-Generated Loss Function.")
        except ImportError:
            pass

    def forward(self, predicted_drift, true_drift_proxy):
        # Check for AI Override
        if self.custom_loss:
            try:
                # Assuming custom loss signature matches: (drift, diffusion, inputs)
                # We need to pass the right args. For this prototype, we pass basic tensors.
                # Just passing drift for now as a stub check.
                return self.custom_loss(predicted_drift, None, None) 
            except Exception as e:
                print(f"Warning: Custom Loss Failed ({e}). Reverting to default.")

        # Default: We enforce that the learned drift matches the observed returns trend
        return self.mse(predicted_drift, true_drift_proxy)

# Training Prototype
def train_model(epochs=200):
    """
    Generator that yields training status for UI integration.
    Yields: (epoch, loss_value)
    """
    load_env()
    print("--- Phase 7: Econophysics Prototype ---")
    
    loader = FinanceLoader()
    
    # Try Real Data First
    t, P = loader.load_real_data()
    if t is None:
        t, P = loader.generate_synthetic_data()
        print(f"Generated {len(P)} synthetic price points.")
    else:
        print(f"Loaded {len(P)} real market points.")
    
    # Prepare inputs: [t, P]
    inputs = torch.cat([t, P], dim=1)
    
    # Proxy for true drift
    returns = (P[1:] - P[:-1]) / loader.dt
    target_drift = torch.cat([returns, torch.zeros(1, 1)], dim=0) # Pad last

    model = NeuralSDE()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Physics-Informed Market Predictor...")
    loss_fn = PhysicsLoss() # Use the custom loss wrapper

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        pred_drift, pred_diff = model(inputs)
        
        # Use Dynamic Physics Loss
        loss = loss_fn(pred_drift, target_drift)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0:
            yield epoch, loss.item()

    print("Training Complete.")

if __name__ == "__main__":
    for epoch, loss in train_model():
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss {loss:.5f}")
