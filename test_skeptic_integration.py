
from skeptic_agent import SkepticAgent

print("TESTING SKEPTIC INTEGRATION...")

# Mocking a "Successful" Translation that is actually hallucinated
# (This is what happens if the Translator's regex check fails but it returns a JSON anyway)
source_text = "The Schrödinger equation describes the wave function of a quantum mechanical system."
bad_mapping = {"fungal_growth": "interest_rate"} # Hallucinated variable 'fungal_growth'
target_domain = "Financial Markets"

skeptic = SkepticAgent()
print(f"\nSOURCE: {source_text}")
print(f"MAPPING: {bad_mapping}")
print("Invoking Skeptic...")

approved, critique = skeptic.critique(source_text, bad_mapping, target_domain)

if not approved:
    print(f"\n✅ SKEPTIC SUCCESS: REJECTED MAPPING. Critique: {critique}")
else:
    print(f"\n❌ SKEPTIC FAILED: APPROVED BAD MAPPING. Critique: {critique}")
