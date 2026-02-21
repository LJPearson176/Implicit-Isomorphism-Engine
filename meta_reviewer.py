
"""
meta_reviewer.py
The Architect Agent.
Responsible for Meta-Cognition: thinking about what the system is thinking about.

Key Capabilities:
1. Meta-Loss Calculation: Scores discoveries based on Novelty (Domain Distance) and Utility (Pragmatism).
2. Topological Gap Analysis: Identifies "Structural Holes" in the graph to direct the Scout.
"""
import os
import glob
import re

class MetaReviewer:
    """
    Evaluates the system's own hypothesis reports to identify high-value science.
    """
    def __init__(self):
        print("--- 🏛️ The Architect: Meta-Reviewer Online ---")
        self.reports_dir = "."
        self.known_pairs = {
            frozenset(["physics", "q-fin"]), # Standard Econophysics
            frozenset(["cs", "q-bio"]),      # Bioinformatics
        }
        
    def load_hypotheses(self):
        """
        Parses all hypothesis_report_*.md files to extract discoveries.
        """
        files = glob.glob(os.path.join(self.reports_dir, "hypothesis_report_*.md"))
        hypotheses = []
        
        for file in files:
            with open(file, 'r') as f:
                content = f.read()
                
            # Regex to extract hypothesis blocks
            # Matches: ### Hypo-[timestamp] ... details ...
            blocks = content.split("### Hypo-")[1:]
            
            for block in blocks:
                try:
                    # Extract fields
                    timestamp = block.split("\n")[0].strip()
                    # Pathway: source $\to$ target
                    pathway_match = re.search(r"\*\*Pathway\*\*: (.*?) \$\\to\$ (.*)", block)
                    score_match = re.search(r"\*\*Resonance Score\*\*: ([\d\.]+)", block)
                    concept_match = re.search(r"\*\*Concept\*\*: (.*)", block)
                    status_match = re.search(r"\*\*Status\*\*: (.*)", block) # e.g. "Proposed Optimization" or "Scheduled..."
                    
                    if pathway_match and score_match:
                        hypotheses.append({
                            "id": timestamp,
                            "source": pathway_match.group(1).strip(),
                            "target": pathway_match.group(2).strip(),
                            "gnn_score": float(score_match.group(1)),
                            "concept": concept_match.group(1) if concept_match else "",
                            "status": status_match.group(1) if status_match else "Unknown"
                        })
                except Exception as e:
                    print(f"Error parsing block in {file}: {e}")
                    
        return hypotheses

    def calculate_domain_distance(self, source, target):
        """
        Heuristic for 'Novelty'. 
        Real impl would use embedding cosine distance (1 - similarity).
        Heuristic: 
        - Same Category (cs.LG -> cs.AI) = 0.1 (Low Novelty)
        - Different Category (cs.LG -> physics.flu-dyn) = 0.8 (High Novelty)
        - Disparate Fields (quant-ph -> q-fin.CP) = 1.0 (Paradigm Shift)
        """
        s_root = source.split('.')[0] # e.g. 'cs' or 'physics'
        t_root = target.split('.')[0]
        
        if s_root == t_root:
            return 0.1 # Incremental
        
        # Check for super-novelty (unseen pairs)
        if frozenset([s_root, t_root]) in self.known_pairs:
            return 0.6 # Interdisciplinary but known
            
        return 1.0 # Radical/Novel

    def score_hypotheses(self, hypotheses):
        """
        Calculate Meta-Loss: L_meta = alpha * (1-Novelty) + beta * Complexity ...
        We want to MAXIMIZE Score (Utility).
        Score = w1 * Novelty + w2 * GNN_Confidence + w3 * Pragmatism
        """
        print(f"\n🔍 Reviewing {len(hypotheses)} Hypotheses...")
        scored = []
        
        for h in hypotheses:
            novelty = self.calculate_domain_distance(h['source'], h['target'])
            confidence = h['gnn_score']
            
            # Pragmatism: Does it claim "Efficiency" or "Optimization"? (Scout Protocol)
            pragmatism = 1.0 if "Optimize" in h['concept'] or "Efficiency" in h['concept'] else 0.5
            
            # Weighted Score
            # We value Novelty and Pragmatism highly.
            meta_score = (0.4 * novelty) + (0.3 * confidence) + (0.3 * pragmatism)
            
            h['meta_score'] = meta_score
            h['novelty'] = novelty
            scored.append(h)
            
        # Sort by Meta-Score
        scored.sort(key=lambda x: x['meta_score'], reverse=True)
        return scored

    def generate_architect_report(self, scored_hypotheses):
        print("\n--- 📋 Architect's Evaluation ---")
        best = scored_hypotheses[:3]
        for h in best:
            print(f"🏆 [Score: {h['meta_score']:.3f}] {h['source']} -> {h['target']}")
            print(f"    Novelty: {h['novelty']} | Confidence: {h['gnn_score']}")
            print(f"    Concept: {h['concept']}")
            print("------------------------------------------------")
            
        # Feedback Loop Suggestion
        # Identify the output of the best performing domains
        top_source = best[0]['source']
        top_target = best[0]['target']
        print(f"\n📢 DIRECTIVE: Focus Scout Agent on intersection of '{top_source}' and '{top_target}'.")
        return best[0]

if __name__ == "__main__":
    reviewer = MetaReviewer()
    hypos = reviewer.load_hypotheses()
    if hypos:
        scored = reviewer.score_hypotheses(hypos)
        reviewer.generate_architect_report(scored)
    else:
        print("No hypotheses found to review. Run 'dream_engine.py' first.")
