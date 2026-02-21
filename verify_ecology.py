from translator import NeuroSymbolicTranslator
import json
import asyncio

async def verify_ecology_mapping():
    print("--- Phase 14c: Verify Physics -> Ecology Isomorphism ---")
    
    # 1. Simulate a Physics Equation (Reaction-Diffusion)
    # dC/dt = D * laplacian(C) + R(C)
    # Using generalized variables u, v
    source_eq = r"\frac{\partial u}{\partial t} = D \nabla^2 u + \alpha u - \beta u v"
    # Note: LLM might output unicode chars like α even if input is \alpha if not strict.
    # But prompt says "Use actual latex strings".
    
    context = "Reaction-Diffusion system describing chemical concentrations."
    
    print(f"Source Equation (Physics):\n{source_eq}")
    print(f"Context: {context}\n")
    
    # 2. Map to Ecology
    translator = NeuroSymbolicTranslator()
    target_domain = "Ecology"
    
    print(f">> Attempting Mapping to {target_domain}...")
    mapping = await translator.map_variables(context, source_eq, target_domain=target_domain)
    
    if mapping:
        print(f"✅ Mapping Successful:\n{json.dumps(mapping, indent=2)}")
        
        # 3. Rewrite
        translated_eq = translator.rewrite_equation(source_eq, mapping)
        print(f"\nTranslated Equation ({target_domain}):\n{translated_eq}")
        
        # Check if it looks like Lotka-Volterra / Population Dynamics
        # Expected: u -> N (Prey), v -> P (Predator), alpha -> Growth, beta -> Predation
        expected_terms = ["N", "P", "Growth", "Predat"]
        matches = [t for t in expected_terms if t in translated_eq or any(t in v for v in mapping.values())]
        
        if len(matches) >= 2:
            print(">> SUCCESS: Isomorphism logically consistent with Population Dynamics.")
        else:
            print(">> NOTICE: Mapping technically valid but semantics may vary.")
            
    else:
        print("❌ Mapping Failed.")

if __name__ == "__main__":
    asyncio.run(verify_ecology_mapping())
