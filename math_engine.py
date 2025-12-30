import torch
from transformers import AutoTokenizer, AutoModel
import re

class MathEmbedder:
    def __init__(self, model_name="tbs17/MathBERT"):
        print(f"Loading MathBERT model: {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            print("MathBERT loaded successfully.")
        except Exception as e:
            print(f"Error loading MathBERT: {e}")
            raise

    def embed_equation(self, latex_str):
        """
        Generates a dense vector embedding for a LaTeX equation string.
        """
        # specialized tokenization or preprocessing could go here
        # standard BERT tokenization for now
        inputs = self.tokenizer(latex_str, return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling of the last hidden state
        # (Batch, SeqLen, HiddenDim) -> (Batch, HiddenDim)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()[0]

    def compute_structural_similarity(self, eq1, eq2):
        v1 = self.embed_equation(eq1)
        v2 = self.embed_equation(eq2)
        
        # Cosine similarity
        similarity = (v1 @ v2) / ((v1**2).sum()**0.5 * (v2**2).sum()**0.5)
        return float(similarity)

if __name__ == "__main__":
    # Test the Embedder
    
    embedder = MathEmbedder()
    
    # Test Case 1: Isomorphism (Navier-Stokes vs Black-Scholes-like)
    # NS (Momentum): Partial u / Partial t + ...
    eq_ns = r"\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla) \mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}"
    
    # Financial (Drift/Diffusion): Partial P / Partial t + ... (Structurally similar)
    eq_fin = r"\frac{\partial P}{\partial t} + (P \cdot \Delta) P = \sigma \Delta^2 P"
    
    # Control (Gravity): F = G m1 m2 / r^2
    eq_grav = r"F = G \frac{m_1 m_2}{r^2}"
    
    sim_bridge = embedder.compute_structural_similarity(eq_ns, eq_fin)
    sim_control = embedder.compute_structural_similarity(eq_ns, eq_grav)
    
    print(f"\nSimilarity [Navier-Stokes <-> Finance]: {sim_bridge:.4f}")
    print(f"Similarity [Navier-Stokes <-> Gravity]: {sim_control:.4f}")
    
    if sim_bridge > sim_control:
        print("SUCCESS: MathBERT detects structural isomorphism better than random baseline.")
    else:
        print("FAILURE: MathBERT failed to distinguish structure.")
