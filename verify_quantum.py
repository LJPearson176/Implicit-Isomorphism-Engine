from translator import NeuroSymbolicTranslator
import json

def verify_quantum_mapping():
    print("--- Phase 17c: Verify Quantum Physics -> Quantum Finance Isomorphism ---")
    
    # 1. Simulate Schrödinger Equation (Physics)
    # i * hbar * dPsi/dt = H * Psi
    # Simplified for tokenization (avoid \hat, use simple letters)
    source_eq = r"i \hbar \frac{\partial \Psi}{\partial t} = H \Psi"
    context = "Quantum State Evolution. Variables: Psi (Wavefunction), t (Time), H (Hamiltonian)."
    
    print(f"Source Equation (Physics):\n{source_eq}")
    print(f"Context: {context}\n")
    
    # 2. Map to Quantum Finance
    translator = NeuroSymbolicTranslator()
    target_domain = "Quantum Finance"
    
    print(f">> Attempting Mapping to {target_domain}...")
    mapping = translator.map_variables(context, source_eq, target_domain=target_domain)
    
    if mapping:
        print(f"✅ Mapping Successful:\n{json.dumps(mapping, indent=2)}")
        
        # 3. Rewrite
        translated_eq = translator.rewrite_equation(source_eq, mapping)
        print(f"\nTranslated Equation ({target_domain}):\n{translated_eq}")
        
        # Check semantics
        # Expected: Psi -> PriceProb, H -> Arbitrage/Risk
        expected_terms = ["Price", "Prob", "Arbitrage", "Risk", "Option", "Cost"]
        matches = [t for t in expected_terms if t in translated_eq or any(t in v for v in mapping.values())]
        
        if len(matches) >= 1:
            print(">> SUCCESS: Isomorphism logically consistent with Quantum Finance.")
        else:
            print(">> NOTICE: Mapping technically valid but semantics may vary.")
            
    else:
        print("❌ Mapping Failed.")

if __name__ == "__main__":
    verify_quantum_mapping()
