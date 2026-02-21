
from translator import NeuroSymbolicTranslator
import json
import asyncio

async def verify_cybernetics():
    print("--- Verifying Phase 32: Cybernetics Capability ---")
    
    # 1. The Source: A standard PID Control Equation (Frequency domain or Time domain)
    # u(t) = K_p e(t) + K_i \int_0^t e(\tau) d\tau + K_d \frac{de(t)}{dt}
    source_eq = r"u(t) = K_p e(t) + K_i \int_{0}^{t} e(\tau) d\tau"
    context = "Feedback Control System for Process Stability"
    target_domain = "Control Theory (Market Stability)"
    
    translator = NeuroSymbolicTranslator()
    
    # 2. Test Variable Mapping
    print(f"\n[Step 1] Mapping Variables for '{target_domain}'...")
    mapping = await translator.map_variables(context, source_eq, target_domain=target_domain)
    print(f"Mapping Result: {json.dumps(mapping, indent=2)}")
    
    # Check for critical control terms
    required_terms = ["u", "e", "K"]
    mapped_terms = [k for k in mapping.keys() if any(t in k for t in required_terms)]
    if not mapped_terms:
         print("❌ FAILED: Did not map key Control variables (u, e, K).")
    else:
         print("✅ PASSED: Mapped Control variables.")

    # 3. Test Code Generation
    print(f"\n[Step 2] Generating Simulation Code...")
    # Apply mapping first
    translated_eq = translator.rewrite_equation(source_eq, mapping)
    print(f"Translated Equation: {translated_eq}")
    
    code = await translator.generate_code(translated_eq, target_domain=target_domain)
    
    print(f"\n[Step 3] Verifying Code Structure...")
    if "simulate_control_loop" in code:
        print("✅ PASSED: 'simulate_control_loop' function found.")
    else:
        print("❌ FAILED: Function 'simulate_control_loop' not found in generated code.")
        
    if "scipy.signal" in code or "numpy" in code:
        print("✅ PASSED: Control libraries (numpy/scipy) detected.")
        
    print("\n--- Generated Code Preview ---")
    print(code[:500] + "...")

if __name__ == "__main__":
    asyncio.run(verify_cybernetics())
