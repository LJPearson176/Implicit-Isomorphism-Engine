import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt

def simulate_quantum_dynamics(steps=100):
    # Define the Hamiltonian matrix H for a 4x4 system (two spin pairs)
    n = 4  # Dimension of the Hilbert space
    H = np.zeros((n, n), dtype=np.complex64)
    
    # Define coupling strength between adjacent spins
    coupling_strength = -1j
    
    # Initialize the Hamiltonian matrix with appropriate couplings
    for i in range(n):
        if (i % 2) == 0:  # Even indices represent "up" spin states
            H[i][i] += 1
            if i < n-1:
                H[i][i+1] += coupling_strength
                H[i+1][i] += np.conjugate(coupling_strength)
    
    # Define initial wavefunction Psi(0): all zeros except the first state is 1 (spin up on root node)
    Psi = np.zeros((n, 1), dtype=np.complex64)
    Psi[0] = 1.0
    
    # Time evolution steps
    Psis = [Psi.copy()]
    
    for step in range(steps):
        # Standard Hamiltonian Evolution
        # U = exp(-iHt)
        # Here H is constant, so U is constant.
        # But if H varies, we compute it each step. H is constant here.
        
        # Proper Matrix Evolution
        U = expm(-1j * H)
        
        # Apply U to the current wavefunction Psi
        Psi = np.dot(U, Psi)
        
        # Normalize the wavefunction (not strictly necessary due to unitarity but included for safety)
        norm = np.linalg.norm(Psi)
        if norm != 0:
            Psi /= norm
        
        Psis.append(Psi.copy())
    
    # Calculate probability densities |Psi|^2 for each state across time steps
    prob_densities = [np.real(np.abs(psi)**2) for psi in Psis]
    
    # Plot the probability densities over time
    # Plot the probability densities over time
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(len(prob_densities)), [p[0] for p in prob_densities], label='Spin 1')
    plt.plot(range(len(prob_densities)), [p[1] for p in prob_densities], label='Spin 2')
    plt.plot(range(len(prob_densities)), [p[2] for p in prob_densities], label='Spin 3')
    plt.plot(range(len(prob_densities)), [p[3] for p in prob_densities], label='Spin 4')
    plt.xlabel('Time Step')
    plt.ylabel('Probability Density')
    plt.title('Probability Densities Over Time (Quantum Finance)')
    plt.legend()
    plt.grid(True)
    return fig

if __name__ == "__main__":
    simulate_quantum_dynamics()
    plt.show()