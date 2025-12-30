
import ollama
import json
import re

class SkepticAgent:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        print(f"🧐 Skeptic Agent initialized ({model_name}). Ready to critique.")

    def critique(self, source_text, variable_map, target_domain):
        """
        Adversarial Review: Checks if the proposed mapping is grounded in the source text.
        Returns: (approved: bool, reason: str)
        """
        
        # 1. Format the critique prompt
        prompt = f"""
        TASK: Peer Review a proposed scientific isomorphism.
        Your job is to be SKEPTICAL. You must reject hallucinations.
        
        SOURCE TEXT (Abstract/Snippet):
        "{source_text[:2000]}..."
        
        PROPOSED VARIABLE MAPPING (Source -> {target_domain}):
        {json.dumps(variable_map, indent=2)}
        
        CRITERIA:
        1. Are the 'source' variables actually mentioned or implied in the text?
        2. Is the mapping semantically logical OR a valid metaphor? (e.g. Temperature -> Volatility is ACCEPTABLE).
        3. Reject ONLY if the mapping invents variables completely unrelated to the source text.
        4. Be permissive of cross-domain analogies. We are looking for Isomorphisms, not literal identity.
        
        RESPONSE FORMAT:
        {"{"}
            "approved": boolean,
            "critique": "string explanation"
        {"}"}
        
        Return ONLY valid JSON.
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            content = response['message']['content']
            
            # Extract JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                result = json.loads(match.group(0))
                return result.get('approved', False), result.get('critique', "No critique provided.")
            else:
                return False, "Skeptic failed to parse its own judgment."
                
        except Exception as e:
            return False, f"Skeptic crashed: {e}"

if __name__ == "__main__":
    # Test Unit
    agent = SkepticAgent()
    
    # Test 1: Good Mapping
    txt = "Brownian motion describes the random movement of particles in a fluid due to collisions."
    good_map = {"particle position": "stock price", "fluid viscosity": "market liquidity"}
    print(f"Test 1 (Good): {agent.critique(txt, good_map, 'Finance')}")
    
    # Test 2: Bad Mapping (Hallucination)
    txt = "The Schrödinger equation describes the wave function of a quantum mechanical system."
    bad_map = {"fungal_growth_rate": "interest_rate", "spore_count": "money_supply"}
    print(f"Test 2 (Bad): {agent.critique(txt, bad_map, 'Finance')}")
