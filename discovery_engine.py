import os
import json
import re
from agent import ResearchAgent
from fetch_source import fetch_source
from translator import EquationExtractor, NeuroSymbolicTranslator
# from translator import VariableMapper, NeuroSymbolicRewriter # DEPRECATED

def find_tex_files(directory):
    tex_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                tex_files.append(os.path.join(root, file))
    return tex_files

def run_discovery_pipeline(query="How does Fluid Turbulence relate to Financial Volatility?", target_domain="Financial Markets"):
    """
    Runs the full pipeline strings -> code.
    Returns a dict with all artifacts for UI/API usage.
    """
    print(f"\n🚀 STARTING DISCOVERY ENGINE [Domain: {target_domain}]")
    print(f"Goal: Find Isomorphism for '{query}'\n")

    result_payload = {
        "success": False,
        "query": query,
        "target_domain": target_domain,
        "paper_title": None,
        "arxiv_id": None,
        "original_eq": None,
        "mapping": {},
        "translated_eq": None,
        "generated_code": None,
        "logs": []
    }
    
    def log(msg):
        print(msg)
        result_payload["logs"].append(msg)

    # 1. Agent Retrieval
    log("--- [Step 1] Agent Retrieval ---")
    agent = ResearchAgent()
    candidates = agent.query(query, top_k=1)
    
    if not candidates:
        log("No candidates found. Aborting.")
        return result_payload

    top_paper = candidates[0]
    arxiv_id = top_paper.get('arxiv_id')
    result_payload["paper_title"] = top_paper.get('title')
    result_payload["arxiv_id"] = arxiv_id
    
    if not arxiv_id or arxiv_id == 'N/A':
        log(f"Top candidate {top_paper['title']} has no ArXiv ID. Aborting source fetch.")
        return result_payload

    log(f"\n>> Top Candidate: {top_paper['title']} ({arxiv_id})")

    # 2. Source Ingestion
    log("\n--- [Step 2] Source Ingestion ---")
    source_dir = fetch_source(arxiv_id)
    if not source_dir:
        log("Failed to fetch source. Aborting.")
        return result_payload
        
    tex_files = find_tex_files(source_dir)
    log(f"Found {len(tex_files)} .tex files.")
    
    # 3. Equation Extraction
    log("\n--- [Step 3] Equation Extraction ---")
    extractor = EquationExtractor()
    all_equations = []
    for tex in tex_files:
        eqs = extractor.extract_from_file(tex)
        all_equations.extend(eqs)
        
    log(f"Extracted {len(all_equations)} equations.")
    
    # Filter/Select
    complex_eqs = [eq for eq in all_equations if len(eq) > 20 and "=" in eq]
    if not complex_eqs:
        log("No complex equations found. Fallback to any.")
        complex_eqs = all_equations

    if not complex_eqs:
         log("No equations found at all.")
         return result_payload

    complex_eqs.sort(key=len, reverse=True)
    target_eq = complex_eqs[0]
    result_payload["original_eq"] = target_eq
    
    log(f"\n>> Candidate Equation (The Law):\n{target_eq[:200]}...")

    # 4. Translation
    log(f"\n--- [Step 4] Mathematical Translation to {target_domain} ---")
    from translator import NeuroSymbolicTranslator
    translator = NeuroSymbolicTranslator()
    
    # 4a. Map Variables (Pass Target Domain)
    mapping = translator.map_variables(top_paper.get('abstract', ''), target_eq, target_domain=target_domain)
    result_payload["mapping"] = mapping
    
    # CIRCUIT BREAKER
    if not mapping:
        log("\n[!] CRITICAL FAILURE: Neuro-Symbolic Translation Failed (Guardrails Active).")
        return result_payload

    log(f">> Derived Isomorphism Mapping:\n{json.dumps(mapping, indent=2)}")
    
    # 4b. Rewrite
    translated_eq = translator.rewrite_equation(target_eq, mapping)
    result_payload["translated_eq"] = translated_eq
    
    log(f"\n==================================================")
    log(f"🧪 DISCOVERY REPORT: {top_paper.get('title')}")
    log(f"--------------------------------------------------")
    log(f"ORIGINAL (Physics):\n{target_eq}")
    log(f"--------------------------------------------------")
    log(f"TRANSLATED ({target_domain}):\n{translated_eq}")
    
    # --- PHASE 28: THE HIVE MIND (Skeptic Review) ---
    import json
    try:
        with open("system_config.json", "r") as f:
            config = json.load(f)
            skeptic_enabled = config.get("skeptic_enabled", True)
    except:
        skeptic_enabled = True

    if skeptic_enabled:
        from skeptic_agent import SkepticAgent
        skeptic = SkepticAgent()
        print("\n--- [Step 4b] Adversarial Peer Review ---")
        
        approved, critique = skeptic.critique(top_paper.get('abstract', ''), mapping, target_domain)
        
        if not approved:
            print(f"❌ SKEPTIC REJECTED MAPPING: {critique}")
            print("Terminating pipeline to prevent hallucination.")
            return result_payload
            
        print(f"✅ SKEPTIC APPROVED: {critique}")
    else:
        print("\n--- [Step 4b] Adversarial Peer Review (SKIPPED by Config) ---")
    # -----------------------------------------------

    print("\n--- [Step 5] Auto-Compilation ---")
    # Pass Target Domain to Code Generator
    code = translator.generate_code(translated_eq, target_domain=target_domain)
    result_payload["generated_code"] = code
    log(f"\n>> AI Generated Code:\n{code}")
    
    from compiler import Compiler
    c = Compiler()
    success = c.compile(code)
    
    if success:
        log(f"\n>> System Status: EQUATION COMPILED for {target_domain}")
        result_payload["success"] = True
    else:
        log("\n>> Compilation Failed.")

    log(f"==================================================\n")
    return result_payload

if __name__ == "__main__":
    import sys
    q = "How does Fluid Turbulence relate to Financial Volatility?"
    d = "Financial Markets"
    if len(sys.argv) > 1:
        q = sys.argv[1]
    if len(sys.argv) > 2:
        d = sys.argv[2]
    run_discovery_pipeline(q, d)
