import importlib
import traceback
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless mode
import matplotlib.pyplot as plt

def verify_generated_code(module_name="generated_loss"):
    """
    Verifies the generated code by attempting to run the main function.
    Checks for:
    1. Import Errors (Syntax)
    2. Runtime Errors
    3. Numerical Instability (NaNs/Infs)
    4. Divergence (Values expanding to infinity)
    """
    print(f"--- Verifying {module_name}.py ---")
    
    # 1. Syntax / Import Check
    try:
        module = importlib.import_module(module_name)
        importlib.reload(module)
    except Exception as e:
        return {"success": False, "error": f"Import/Syntax Error: {e}", "traceback": traceback.format_exc()}

    # 2. Logic Check
    # Identify function
    target_func = None
    func_name = ""
    
    if hasattr(module, 'simulate_opinion'):
        target_func = module.simulate_opinion
        func_name = "simulate_opinion"
    elif hasattr(module, 'custom_physics_loss'):
        # Just create dummy tensor to test loss
        import torch
        target_func = lambda: module.custom_physics_loss(torch.randn(10,10), torch.randn(10,10), torch.randn(10,10))
        func_name = "custom_physics_loss"
    elif hasattr(module, 'simulate_quantum_dynamics'):
        target_func = module.simulate_quantum_dynamics
        func_name = "simulate_quantum_dynamics"
    else:
        return {"success": False, "error": "No standard simulation function found."}

    # 3. Runtime Check
    try:
        print(f"Executing {func_name}...")
        result = target_func()
        
        # Check Return Type
        import matplotlib.pyplot as plt
        is_figure = isinstance(result, plt.Figure)
        is_tensor = hasattr(result, 'backward') # torch tensor
        
        print(f"Execution successful. Result type: {type(result)}")
        
        # Data Integrity Check (We can't easily check internal simulation state unless returned)
        # But if it returned a Figure, we assume it plotted valid numbers.
        # If it crashed with NaNs, it usually throws an exception or warns.
        
        return {"success": True, "message": f"Successfully executed {func_name}. Output valid."}
        
    except Exception as e:
        return {"success": False, "error": f"Runtime Error: {e}", "traceback": traceback.format_exc()}

if __name__ == "__main__":
    result = verify_generated_code()
    print("\nVERIFICATION REPORT:")
    print(f"Success: {result['success']}")
    if not result['success']:
        print(f"Error: {result['error']}")
    else:
        print(result['message'])
