"""
compiler.py
The Synthesis Agent.
Converts abstract mathematical plans into executable Python artifacts.

Key Capabilities:
1. compile: Writes code to disk (`generated_loss.py`) and handles imports.
2. auto_synthesize: The full "Agentic Loop" -> Load Edges -> Translate -> Compile -> Verify -> Fix.
3. compile_isomorphism: Helper for the Scout Protocol to compile rapid prototypes.
"""
import os
import json
import random
import time
from translator import NeuroSymbolicTranslator
from data_engine import DataEngine

class Compiler:
    def __init__(self, output_file="generated_loss.py"):
        self.output_file = output_file
        self.translator = NeuroSymbolicTranslator()
        # No local DataEngine needed if we just use ArXiv data, 
        # but we might need fetching if abstract is missing.
    
    def compile(self, code_string):
        """
        Writes the generated code string to a python module file.
        Ensures necessary imports are present.
        """
        print(f"Compiling AI Code to {self.output_file}...")
        
        # Ensure imports (Simple heuristic merge)
        base_imports = ["import torch", "import torch.nn as nn", "import numpy as np"]
        
        lines = code_string.split('\n')
        final_lines = []
        
        # Add base imports
        for imp in base_imports:
            final_lines.append(imp)
            
        for line in lines:
            if "import" in line and any(b in line for b in base_imports):
                continue # Skip dupes
            final_lines.append(line)
            
        full_code = "\n".join(final_lines)
        
        try:
            with open(self.output_file, "w") as f:
                f.write(full_code)
            print(f"Compilation Successful: {self.output_file} created.")
            return True
        except Exception as e:
            print(f"Compilation Failed: {e}")
            return False

    def compile_isomorphism(self, bridge_edge):
        """
        Compiles a specific isomorphism bridge into executable code.
        Used by the Scout Protocol.
        """
        print(f"Compiling Bridge: {bridge_edge['source_equation_name']} -> {bridge_edge['target_domain']}")
        
        # 1. Prepare "Equation" string
        # Since we might not have a raw LaTeX string, we use the name and mapping as the 'equation' context for the LLM
        eq_context = f"{bridge_edge['source_equation_name']} (Mapping: {bridge_edge['variable_mapping']})"
        
        # 2. Generate Code
        # We pass the equation name + mapping as the 'translated_eq' to guide the LLM
        code = self.translator.generate_code(eq_context, target_domain=bridge_edge['target_domain'])
        
        # 3. Compile
        if code:
             # Use a custom filename for scout solutions
             self.output_file = "generated_scout_solution.py"
             if self.compile(code):
                 return self.output_file
        
        return None

    def auto_synthesize(self, edges_file="bridge_edges.json"):
        """
        The Agentic Workflow:
        1. Load Bridge Edges
        2. Pick High-Value Candidate (Physics -> Finance/Social/Bio)
        3. Extract Equation (Simulated/Heuristic or simple prompt if extraction unavailable)
        4. Translate
        5. Compile
        """
        print("\n--- Starting Auto-Synthesis Agent ---")
        
        if not os.path.exists(edges_file):
            print(f"Error: {edges_file} not found. Run visualize_bridge.py first.")
            return

        with open(edges_file, 'r') as f:
            edges = json.load(f)
            
        print(f"Loaded {len(edges)} bridge candidates.")
        
        # Filter for interesting bridges (Source=Physics, Target!=Physics)
        candidates = [
            e for e in edges 
            if (e['source_silo'] == 'Physics' or e['source_silo'] == 'Quantum') 
            and e['target_silo'] != 'Physics' and e['target_silo'] != 'AI'
        ]
        
        if not candidates:
            print("No high-value Physics bridges found. Fallback to random.")
            candidates = edges
            
        # Pick one
        target = random.choice(candidates)
        print(f"Selected Bridge Edge (Sim: {target['similarity']:.3f}):")
        print(f"  Source: {target['source_arxiv']} ({target['source_silo']})")
        print(f"  Target: {target['target_arxiv']} ({target['target_silo']})")
        
        # Fetch Content (Using ArXiv Lib directly for speed)
        import arxiv
        print("Fetching paper details...")
        client = arxiv.Client()
        search = arxiv.Search(id_list=[target['source_arxiv']])
        results = list(client.results(search))
        
        if not results:
            print("Error: Could not fetch source paper.")
            return
            
        paper = results[0]
        title = paper.title
        abstract = paper.summary
        print(f"  Title: {title}")
        
        # Heuristic Extraction (Prompt-based since we don't have full PDF text here yet)
        # We ask the LLM to "Hallucinate" the likely equation based on the Title/Abstract 
        # (This is a valid strategy for "Discovery" on abstract-only data)
        print("Extracting/Inferring Equation from Abstract...")
        
        # Translation
        target_domain_map = {
            "Finance": "Financial Markets",
            "Bio": "Ecology (Population Dynamics)",
            "Social": "Social Science (Opinion Dynamics)",
            "Quantum": "Quantum Finance"
        }
        target_domain = target_domain_map.get(target['target_silo'], "Financial Markets")
        
        # We use a special extractor prompt
        equation_prompt = f"Extract the core mathematical equation (LaTeX) that describes '{title}'. If not explicitly stated, infer the standard physical law associated with this topic (e.g. Navier-Stokes for Turbulence, Schrodinger for Quantum). Output LaTeX only."
        
        # Using Translator's internal LLM logic (we need to expose it or reuse map_variables?)
        # Let's reuse map_variables but with a hacked context to get the equation first? 
        # No, map_variables expects an equation.
        # Let's just hardcode a "Discovery" prompt here using Ollama directly for flexibility.
        import requests
        
        def query_ollama(prompt):
            try:
                resp = requests.post("http://127.0.0.1:11434/api/chat", json={
                    "model": "deepseek-r1:7b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                })
                return resp.json()['message']['content']
            except:
                return "dS = mu S dt + sigma S dW" # Fallback
        
        extracted_eq = query_ollama(equation_prompt)
        print(f"Inferred Equation: {extracted_eq[:50]}...")
        
        # Now Execute Pipeline
        context = f"Paper: {title}. Abstract: {abstract}."
        
        print(f"Translating to {target_domain}...")
        mapping = self.translator.map_variables(context, extracted_eq, target_domain=target_domain)
        
        if mapping:
            print("Mapping found. Generating Code...")
            translated_eq = self.translator.rewrite_equation(extracted_eq, mapping)
            code = self.translator.generate_code(translated_eq, target_domain=target_domain)
            
            if code:
                self.compile(code)
                print(">> Initial Compilation Complete. Running Verification...")
                
                # Verify
                from verifier import verify_generated_code
                check = verify_generated_code()
                
                if check['success']:
                    print(">> VERIFICATION PASSED: Code is stable.")
                else:
                    print(f">> VERIFICATION FAILED: {check['error']}")
                    print(">> Engaging Self-Correction Loop...")
                    
                    # Fix
                    # We can ask LLM to fix syntax errors
                    fixed_code = self.translator.fix_code(code, check['error'], target_domain)
                    if fixed_code:
                         self.compile(fixed_code)
                         print(">> CORRECTION APPLIED. Re-verifying...")
                         check2 = verify_generated_code()
                         if check2['success']:
                             print(">> SELF-CORRECTION SUCCESSFUL.")
                         else:
                             print(">> SELF-CORRECTION FAILED. Human review needed.")
                    else:
                        print(">> Fixer could not generate a solution.")

            else:
                print("Code Generation failed.")
        else:
            print("Mapping failed.")

if __name__ == "__main__":
    c = Compiler()
    c.auto_synthesize()
