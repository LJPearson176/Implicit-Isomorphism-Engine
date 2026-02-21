import torch
import numpy as np
import json
from math_engine import MathEmbedder
from translator import NeuroSymbolicTranslator
import asyncio

# Calibration Set: A subset of standard Physics laws from AI Feynman (Feynman Lectures)
FEYNMAN_EQUATIONS = [
    # I.6.20: Gaussian (Normal Distribution)
    {"id": "I.6.20", "latex": r"e^{-\frac{\theta^2}{2\sigma^2}} / \sqrt{2\pi\sigma^2}", "name": "Gaussian"},
    # I.12.11: Momentum Conservation / Force
    {"id": "I.12.11", "latex": r"F = q(E + v \times B)", "name": "Lorentz Force"},
    # I.15.10: Momentum
    {"id": "I.15.10", "latex": r"p = \frac{m v}{\sqrt{1 - v^2/c^2}}", "name": "Relativistic Momentum"},
    # I.40.1: Diffusion (Brownian Motion) / Heat Equation
    {"id": "I.40.1", "latex": r"\frac{\partial n}{\partial t} = D \frac{\partial^2 n}{\partial x^2}", "name": "Diffusion 1D"},
    # II.21.32: Voltage (Electrostatics)
    {"id": "II.21.32", "latex": r"\phi = \frac{q}{4\pi\epsilon_0 r}", "name": "Coulomb Potential"}
]

# Target Set: Financial Equations (Mocked from q-fin.ST papers)
# Some are isomorphisms of the Feynman set, others are noise.
FINANCIAL_EQUATIONS = [
    # Isomorph to Diffusion (I.40.1): Black-Scholes PDE (simplified heat eq form)
    {"id": "FIN.BS.1", "latex": r"\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS \frac{\partial V}{\partial S} - rV = 0", "type": "isomorph", "match_id": "I.40.1"},
    # Isomorph to Gaussian (I.6.20): Return Distribution
    {"id": "FIN.DIST.1", "latex": r"\frac{1}{\sqrt{2\pi s^2}} \exp\left(-\frac{(x - \mu)^2}{2s^2}\right)", "type": "isomorph", "match_id": "I.6.20"},
    # Noise: Definition of ROI
    {"id": "FIN.NOISE.1", "latex": r"ROI = \frac{P_f - P_i}{P_i}", "type": "noise"},
    # Noise: Simple Accounting Identity
    {"id": "FIN.NOISE.2", "latex": r"A = L + E", "type": "noise"}
]

async def run_benchmark():
    print("Loading MathEmbedder (The 'Eyes')...")
    embedder = MathEmbedder() # Loads MathBERT
    
    print("\n--- Running AI Feynman Benchmark ---")
    
    correct_retrieval = 0
    total_isomorphs = 0
    
    for f_eq in FEYNMAN_EQUATIONS:
        # We only care about Feynman eqs that actually have a partner in our Finance set
        matches = [fin for fin in FINANCIAL_EQUATIONS if fin.get('match_id') == f_eq['id']]
        if not matches:
            continue
            
        total_isomorphs += 1
        target_match = matches[0]
        
        print(f"\nQuery: {f_eq['name']} ({f_eq['latex']})")
        
        # Rank all Financial Eqs by Structural Similarity
        scores = []
        for fin_eq in FINANCIAL_EQUATIONS:
            sim = embedder.compute_structural_similarity(f_eq['latex'], fin_eq['latex'])
            scores.append((sim, fin_eq))
            
        # Sort desc
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Top-1 Accuracy Check
        top_match = scores[0][1]
        print(f"  Top Match ({scores[0][0]:.4f}): {top_match['id']} - {top_match['latex']}")
        
        if top_match['id'] == target_match['id']:
            print("  ✅ SUCCESS: Correct Structural Isomorphism retrieved.")
            correct_retrieval += 1
        else:
            print(f"  ❌ FAILURE: Expected {target_match['id']}.")

    acc = correct_retrieval / total_isomorphs if total_isomorphs > 0 else 0
    print(f"\n--- Benchmark Results ---")
    print(f"Retrieval Accuracy: {acc*100:.1f}%")
    
    # 2. Translation Quality Check (Qualitative/LLM)
    # Check if VariableMapper can map Diffusion -> BlackScholes
    print("\n--- Translation Check (Diffusion -> BS) ---")
    source = FEYNMAN_EQUATIONS[3]['latex'] # Diffusion
    target_domain = "Quantitative Finance (Black-Scholes)"
    
    mapper = NeuroSymbolicTranslator()
    mapping = await mapper.map_variables(context="Diffusion", equation_tex=source, target_domain=target_domain)
    
    print(f"Mapping Proposed: {json.dumps(mapping, indent=2)}")
    # Heuristic check: Did it map 'n' (concentration) to something like 'V' (Value) or 'P' (Price)?
    # And 'D' (Diffusion coeff) to 'sigma' (Volatility)?
    
    relevant_keys = [k for k in mapping.keys() if 'n' in k or 'D' in k]
    if relevant_keys:
         print("✅ Variable Mapper found relevant concepts.")
    else:
         print("⚠️ Variable Mapper might have missed key variables.")

if __name__ == "__main__":
    from verify_setup import load_env
    load_env()
    asyncio.run(run_benchmark())
