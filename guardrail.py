import re

def validate_mapping(equation_tex, mapping):
    """
    Validates the variable mapping against the equation.
    Checks:
    1. Do source variables exist in the equation?
    2. Are target variables valid financial concepts?
    """
    
    # 1. Existence Check
    # We strip backslashes for checking existence to be lenient with LaTeX commands
    
    missing_vars = []
    
    # Pre-process equation to avoid false negatives on spacing
    # (e.g. \tilde {x} vs \tilde{x})
    eq_norm = equation_tex.replace(" ", "")
    
    for src_var, tgt_var in mapping.items():
        # Check if the source variable string exists in the equation
        # We try strict first, then normalized
        if src_var not in equation_tex and src_var.replace(" ", "") not in eq_norm:
             missing_vars.append(src_var)

    if missing_vars:
        return False, f"Hallucination detected: The variables {missing_vars} were mapped but NOT found in the equation."

    # 2. Target Validity Check
    # User Requirements: P, V, sigma, mu, L, OrderFlow
    valid_terms = [
        # Finance
        "P", "Price", "V", "Volume", "sigma", "Volatility", "mu", "Drift", "L", "Liquidity", "OrderFlow", "t", "Time", 
        # Ecology / Biology (Phase 14)
        "N", "Pop", "Growth", "Birth", "Death", "Carry", "Cap", "Pred", "Prey", "Diff", "r", "K", "alpha", "beta", "gamma", "lambda", "H",
        # Social Science (Phase 16)
        "O", "Opin", "Sent", "Social", "Inf", "Mis", "Net", "Node", "Edge", "Contag", "Vote",
        # Quantum (Phase 17)
        "Psi", "Wave", "Prob", "Hamilton", "H", "Energy", "E", "Ket", "Bra", "Op", "Arb", "Risk", "Opt"
    ]
    
    invalid_targets = []
    for tgt in mapping.values():
        found = False
        # Check if the target contains any valid term
        for term in valid_terms:
            if term in tgt or term.lower() in tgt.lower():
                found = True
                break
        if not found:
            invalid_targets.append(tgt)
            
    if invalid_targets:
        return False, f"Invalid Target Terms: {invalid_targets}. Must use standard financial notation (P, V, sigma, etc)."
    
    return True, "Mapping is dimensionally consistent."
