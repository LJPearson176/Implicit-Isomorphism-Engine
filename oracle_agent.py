
import ollama
import glob
import os

class OracleAgent:
    def __init__(self, model_name="deepseek-r1:7b"):
        self.model_name = model_name
        print(f"🔮 Oracle Agent initialized ({model_name}). Reading Akasha Records...")
        self.context = self.load_context()

    def load_context(self):
        """
        Loads relevant artifacts to form the 'Long Term Memory' of the Oracle.
        """
        context = ""
        
        # 1. Load Recent Hypotheses (Dreams)
        reports = glob.glob("hypothesis_report_*.md")
        reports.sort(reverse=True)
        if reports:
            latest = reports[0] # Just the latest for now to save context tokens
            with open(latest, "r") as f:
                context += f"\n--- LATEST DREAM JOURNAL ({latest}) ---\n{f.read()}\n"
                
        # 2. Load Generated Code (The Law)
        if os.path.exists("generated_loss.py"):
            with open("generated_loss.py", "r") as f:
                context += f"\n--- GENERATED SIMULATION CODE (generated_loss.py) ---\n{f.read()[:2000]}...\n" # Truncate if too long
                
        # 3. Load System Config
        if os.path.exists("system_config.json"):
             with open("system_config.json", "r") as f:
                context += f"\n--- SYSTEM CONFIG ---\n{f.read()}\n"
        
        return context

    def consult(self, query):
        """
        Answers a user query based on the system's internal state.
        """
        prompt = f"""
        YOU ARE THE ORACLE OF THE ANTIGRAVITY ENGINE.
        Your goal is to explain the system's scientific discoveries and internal logic to the user.
        
        CONTEXT (The System's Internal State):
        {self.context}
        
        USER QUERY:
        "{query}"
        
        INSTRUCTIONS:
        1. Answer based *strictly* on the provided context if possible.
        2. If the user asks about a specific hypothesis (e.g. Mycology), refer to the Dream Journal.
        3. If the user asks about the simulation, refer to the Generated Code.
        4. Be mystical but scientifically precise. You are an AI Scientist.
        """
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"The Oracle is silent. (Error: {e})"

if __name__ == "__main__":
    oracle = OracleAgent()
    print(oracle.consult("What was the last scientific epiphany you had?"))
