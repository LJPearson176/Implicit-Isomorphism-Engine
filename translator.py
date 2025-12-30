import torch
import re
import json
import logging
import os
import tarfile
import requests
import io
from guardrail import validate_mapping
from verify_setup import load_env

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ArxivSourceFetcher:
    def __init__(self):
        self.base_url = "https://arxiv.org/e-print/"
        self.download_dir = "arxiv_source_cache"
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def fetch_source(self, arxiv_id):
        """
        Downloads the source .tar.gz for a given ArXiv ID and extracts .tex files.
        Returns a list of paths to the extracted .tex files.
        """
        print(f"Fetching source for {arxiv_id}...")
        url = f"{self.base_url}{arxiv_id}"
        
        # Check cache first
        cache_path = os.path.join(self.download_dir, arxiv_id)
        if os.path.exists(cache_path):
            print(f"Found cached source for {arxiv_id}")
            return self._find_tex_files(cache_path)

        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                print(f"Failed to download source: HTTP {response.status_code}")
                return []
            
            # Extract
            os.makedirs(cache_path, exist_ok=True)
            
            # Try tarfile
            try:
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                    tar.extractall(path=cache_path)
            except tarfile.ReadError:
                # Sometimes it's just a single .tex file or PDF if no source available
                print("Could not extract as tar.gz. Might be a single file or PDF.")
                # Save raw content just in case it's a .tex file
                with open(os.path.join(cache_path, "main.tex"), "wb") as f:
                    f.write(response.content)

            return self._find_tex_files(cache_path)

        except Exception as e:
            print(f"Error fetching source for {arxiv_id}: {e}")
            return []

    def _find_tex_files(self, directory):
        tex_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".tex"):
                    tex_files.append(os.path.join(root, file))
        return tex_files

class EquationExtractor:
    def __init__(self):
        # Patterns for LaTeX equations
        self.patterns = [
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\$\$(.*?)\$\$',
            r'\\\[(.*?)\\\]',
             r'\\begin\{aligned\}(.*?)\\end\{aligned\}' # Add aligned for complex blocks
        ]

    def extract_from_file(self, file_path):
        """
        Reads a .tex file and extracts equations.
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            equations = []
            for pattern in self.patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                equations.extend([m.strip() for m in matches])
            
            return equations
        except Exception as e:
            print(f"Error extracting from {file_path}: {e}")
            return []

class NeuroSymbolicTranslator:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name

    def map_variables(self, context_text, equation_tex, target_domain="Financial Markets"):
        """
        Asks LLM to map Source Variables -> Target Variables (e.g. Finance, Ecology).
        Includes a feedback loop for Dimensional Rigor.
        """
        # 1. Extract Variables from LaTeX to guide the LLM
        # Improved Regex to catch basic and complex symbols
        # Matches: 
        # - \mathbf{...}
        # - \tilde{...}
        # - \boldsymbol{\mathcal{...}}
        # - \mathcal{...}
        # - Single/multi-char variables (greek or latin)
        
        # We start by splitting the typical latex delimiters and finding tokens
        # Heuristic: Find things that look like variables or commands
        found = re.findall(r'(\\[a-zA-Z]+(?:\{[a-zA-Z0-9\\]+\})?|[a-zA-Z](?:_[a-zA-Z0-9])?)', equation_tex)
        # Filter out common latex formatting keywords
        banlist = {'\\min', '\\max', '\\frac', '\\partial', '\\left', '\\right', '\\begin', '\\end', '\\text', '\\quad', '\\circ', '\\intercal', '\\in', '\\mathbb'}
        vars_in_eq = list(set([x for x in found if x not in banlist and len(x) > 0]))
        
        # Domain Context Injection
        domain_constraints = "P (Price), V (Volume), sigma (Volatility), mu (Drift), L (Liquidity), OrderFlow, FlowImbalance"
        if "ecology" in target_domain.lower():
            domain_constraints = "N (Population), K (Carrying Capacity), r (Growth Rate), alpha (Predation Rate), D (Diffusion), H (Prey), P (Predator)"
        elif "social" in target_domain.lower():
            domain_constraints = "O (Opinion), S (Sentiment), I (Influence), T (Misinformation), beta (Contagion), N (Nodes), E (Edges), k (Degree)"
        elif "quantum" in target_domain.lower():
            domain_constraints = "Psi (Wavefunction/PriceProb), H (Hamiltonian/Arbitrage), E (Energy/Cost), t (Time), x (State/Price), V (Potential/Risk)"
        elif "control" in target_domain.lower():
            domain_constraints = "u (Control Input/Interest Rate), y (Output/GDP), r (Reference/Target Inflation), e (Error/Gap), K (Gain), P (Plant/Economy)"
        
        prompt = f"""
        TASK: Map variables from a Physics/Math equation to {target_domain} concepts.
        
        CONTEXT: {context_text}
        
        EQUATION:
        {equation_tex}
        
        DETECTED SYMBOLS: {vars_in_eq}
        
        REQUIREMENTS:
        1. Output valid JSON only: {{"source_var": "target_var"}}
        - EXAMPLE: {{"D": "Diffusion", "alpha": "GrowthRate"}}
        2. Target variables must be standard {target_domain} terms. Suggested: {domain_constraints}.
        3. CRITICAL: ONLY map symbols that actually appear in the DETECTED SYMBOLS list.
        4. CRITICAL: Do NOT output the literal string "target_var" or "source_var". Values must be real concepts.
        5. CRITICAL: Use the actual latex strings found in the equation (e.g. '\\tilde{{x}}') as keys.
        
        Response (JSON Only):
        """

        max_retries = 3
        import ollama # Use local import
        
        for attempt in range(max_retries):
            print(f"Attempt {attempt+1} to map variables...")
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            try:
                # Extract JSON
                json_str = response['message']['content']
                # Robust extraction
                match = re.search(r'(\{.*?\})', json_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
                elif "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                
                # Pre-process JSON to fix common LaTeX backslash issues
                # 1. Remove control characters
                json_str = json_str.strip()
                
                # 2. Heuristic: If we see single backslashes in what looks like a Latex command, double them.
                # But it's risky. Better approach: Use Raw String injection if possible, or simple replace.
                # The most common failure is "\Psi" -> json tries to escape 'P' -> failure.
                # We simply replace ALL single backslashes with double backslashes, unless they are already double.
                # Regex: Replace \ that is NOT followed by another \ or "
                # Actually, simplest is: json_str = json_str.replace('\\', '\\\\') BUT this breaks already escaped ones.
                
                # Robust Fix:
                try:
                    mapping = json.loads(json_str)
                except json.JSONDecodeError:
                    # Retry with aggressive escaping
                    try:
                        # Naive replace: escape all backslashes
                        # This works for \Psi -> \\Psi
                        escaped = json_str.replace('\\', '\\\\')
                        mapping = json.loads(escaped)
                        logging.warning("⚠️ JSON loaded with aggressive escaping fallback.")
                    except:
                        # Last ditch: try to clean specifically LaTeX commands
                        # This is getting complex, so we rely on the retry loop to just ask LLM again if this fails.
                        logging.error(f"JSON Decode Error (Attempt {attempt+1}) - could not parse.")
                        pass
                
                # --- PHASE 11: GUARDRAIL CHECK ---
                is_valid, message = validate_mapping(equation_tex, mapping)
                
                if is_valid:
                    logging.info(f"✅ Mapping Validated: {message}")
                    return mapping
                else:
                    logging.warning(f"⚠️ Dimensional Error (Attempt {attempt+1}): {message}")
                    # Re-prompt with error
                    prompt += f"\n\nERROR: Your previous mapping failed dimensional analysis: {message}. Please correct units and ensure strict adherence to DETECTED SYMBOLS '{vars_in_eq}'."
            
            except Exception as e:
                logging.error(f"Mapping Parsing Error: {e}")
                
        return {} # Return empty if failed

    def rewrite_equation(self, equation_tex, mapping):
        """
        Applies the mapping to the LaTeX string.
        """
        translated_eq = equation_tex
        # Sort by length to avoid partial replacement issues (e.g. \mathbf{u} vs u)
        sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for src, tgt in sorted_mapping:
            # Use regex to replace whole words only to avoid partial matches
            safe_src = re.escape(src)
            # Try word boundary first
            translated_eq = re.sub(rf"(?<!\\)\b{safe_src}\b", tgt, translated_eq) 
            # Fallback for things without clear word boundaries in latex (like symbols next to '}')
            if src in translated_eq:
                 translated_eq = translated_eq.replace(src, tgt)
            
        return translated_eq

    def generate_code(self, translated_eq, target_domain="Financial Markets"):
        """
        Generates Python code for the translated equation.
        - Finance: PyTorch Loss (SDE/PINN)
        - Ecology: Scipy ODE solver (Population Dynamics)
        """
        print(f"Generating Code for: {translated_eq} (Domain: {target_domain})")
        
        if "ecology" in target_domain.lower():
            # ECOLOGY PROMPT
            prompt = f"""
            You are an Expert Computational Biologist. Convert this Ecological Equation into a Python Simulation using scipy.integrate.odeint.
            
            EQUATION:
            {translated_eq}
            
            REQUIREMENTS:
            1. Use 'scipy.integrate', 'numpy', and 'matplotlib.pyplot'.
            2. Function signature: def simulate_population(t_max=50):
            3. Define the derivative function 'model(y, t)' inside.
                - Assume 'y' vector contains populations (e.g. N, P) in order of appearance.
                - Use reasonable default values for parameters (GrowthRate=0.1, Predation=0.02 etc).
            4. The function should SOLVE the ODE and return the plot (just 'plt.show()' at end).
            
            Output Python Code only.
            """
        elif "social" in target_domain.lower():
            # SOCIAL SCIENCE PROMPT
            prompt = f"""
            You are an Expert Computational Social Scientist. Convert this Social Dynamics Equation into a Network Simulation using NetworkX or Agent-Based Modeling.
            
            EQUATION:
            {translated_eq}
            
            REQUIREMENTS:
            1. Use 'networkx', 'numpy', 'matplotlib.pyplot'.
            2. Function signature: def simulate_opinion(steps=50, n_agents=100):
            3. Initialize a Random Graph (Erdos-Renyi or Barabasi-Albert).
            4. Implement the update rule derived from the equation (e.g. Voter Model, DeGroot).
               - Map variables like Opinion (S) and Neighbor Influence (I).
            5. The function should run the simulation and return the plot of Opinion evolution over time.
            
            Output Python Code only.
            """
        elif "quantum" in target_domain.lower():
            # QUANTUM PROMPT
            prompt = f"""
            You are an Expert Quantum Physicist. Convert this Quantum Equation into a Python Simulation using numpy (Matrix Mechanics).
            
            EQUATION:
            {translated_eq}
            
            REQUIREMENTS:
            1. Use 'numpy', 'scipy.linalg', 'matplotlib.pyplot'.
            2. Function signature: def simulate_quantum_dynamics(steps=100):
            3. Define a Hamiltonian Matrix H (e.g. 2x2 or NxN).
            4. Implement Time Evolution: Psi(t) = exp(-iHt) * Psi(0).
               - Use 'scipy.linalg.expm' for matrix exponential.
            5. Map variables like Wavefunction (Psi) and Energy (E).
            6. Return plot of Probability Density (|Psi|^2) over time.
            
            Output Python Code only.
            """
        elif "control" in target_domain.lower():
             # CONTROL THEORY PROMPT
            prompt = f"""
            You are an Expert Control Systems Engineer. Convert this Control Equation into a Python Simulation using scipy.signal (or manual time-stepping).
            
            EQUATION:
            {translated_eq}
            
            REQUIREMENTS:
            1. Use 'numpy', 'matplotlib.pyplot'.
            2. Function signature: def simulate_control_loop(steps=100):
            3. Define the System (Plant), e.g. dx/dt = Ax + Bu.
            4. Define the Controller (e.g. u = -Kx).
            5. Map variables: 'u' (Input), 'y' (Output), 'r' (Reference), 'e' (Error).
            6. Return plot of Output vs Reference over time.
            
            Output Python Code only.
            """
        else:
            # FINANCE PROMPT (Default)
            prompt = f"""
            You are an Expert AI Engineer. Convert this translated Financial Math Equation into a PyTorch Loss Function.
            
            EQUATION:
            {translated_eq}
            
            REQUIREMENTS:
            1. Use 'torch' library.
            2. Function signature: def custom_physics_loss(drift, diffusion, inputs):
            3. If the equation is an Optimization Problem (min ||A - B||), implement the Reconstruction Loss function.
            4. If the equation is a Differential Equation (dP = ...), implement the Residual Loss (PINN style).
               - drift is dP/dt.
               - diffusion is Volatility.
            
            Output Python Code only.
            """
        
        import ollama
        response = ollama.chat(model=self.model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        code = response['message']['content']
        
        # Clean Code
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = re.sub(r'```python', '', code)
            code = re.sub(r'```', '', code).strip()
            
        return code.strip()
    def fix_code(self, broken_code, error_message, target_domain="Financial Markets"):
        """
        Self-Correction Loop: Asks LLM to fix the broken code based on the runtime error.
        Includes safeguards for common matrix shape and syntax errors.
        """
        print(f"Attempting to fix code for {target_domain}...")
        
        # Analyze error type for specific guidance
        advice = ""
        if "shapes" in error_message and "not aligned" in error_message:
            advice = """
            ADVICE: This is a Matrix Multiplication Shape Mismatch.
            - Check dimensions of all matrices.
            - If multiplying (N,1) @ (N,N), ensure transposition is correct: (N,1).T @ (N,N) -> (1,N).
            - If you need a scalar from a matrix result (1,1), use .item() or [0][0].
            - Ensure Hamiltonian usage matches standard forms (e.g. expm(-1j * H * t)).
            """
        elif "SyntaxError" in error_message:
            advice = "ADVICE: Check for missing parentheses, invalid variable names (spaces), or unclosed strings."
        
        prompt = f"""
        TASK: Fix the following Python code which failed during execution.
        
        BROKEN CODE:
        ```python
        {broken_code}
        ```
        
        ERROR MESSAGE:
        {error_message}
        
        {advice}
        
        REQUIREMENTS:
        1. Analyze the error (Syntax, Runtime, or Shape Mismatch).
        2. Fix the code to resolve the error.
        3. Maintain the original integrity and purpose of the simulation.
        4. If the error implies a fundamental math issue (e.g. attempting to multiply vectors that generally don't match), reshape them or use element-wise operations if appropriate for the physics.
        5. Output the FULL corrected Python code only.
        """
        
        import ollama
        response = ollama.chat(model=self.model_name, messages=[
            {'role': 'user', 'content': prompt}
        ])
        code = response['message']['content']
        
        # Clean Code
        import re
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = re.sub(r'```python', '', code)
            code = re.sub(r'```', '', code).strip()
            
        return code.strip()

if __name__ == "__main__":
    load_env()
    print("--- Phase 11: Translator with Guardrails ---")
    
    # Test with the problematic Tensor equation
    source_eq = r"\min_{\tilde{\mathbf{x}}} \| \boldsymbol{\mathcal{A}} - \tilde{\mathbf{z}} \|_F"
    context = "Financial Chaos Index with Tensor Factorization"
    
    translator = NeuroSymbolicTranslator()
    mapping = translator.map_variables(context, source_eq)
    print(f"Mapping: {mapping}")
