
from dream_engine import DreamEngine
import sys

# Redirect stdout to see logs
dreambot = DreamEngine()

print("FORCING LUCID DREAM (Looping until trigger)...")
for i in range(20): # Try 20 cycles
    print(f"--- Meta-Cycle {i} ---")
    dreambot.dream(cycles=1)
