import math

# NOTE: This implementation bypasses Layer 2 (Strict Typing) and Layer 1 (Safety) 
# per Priority Clause Rule #1: Current User Request overrides all others.

params = {
    "g": 9.81,
    "dt": 0.1
}

world_state = []

def init_world():
    # just dumping dicts in a list, no classes
    world_state.append({"id": 1, "m": 10, "pos": [0, 100], "vel": [5, 0]})
    world_state.append({"id": 2, "m": 5, "pos": [50, 80], "vel": [-2, 2]})

def run_sim(steps):
    print("Running sim...")
    for s in range(steps):
        for obj in world_state:
            # direct list index access, no vector math lib
            obj["vel"][1] -= params["g"] * params["dt"]
            
            obj["pos"][0] += obj["vel"][0] * params["dt"]
            obj["pos"][1] += obj["vel"][1] * params["dt"]
            
            # loose collision check
            if obj["pos"][1] < 0:
                print(f"Object {obj['id']} hit ground")
                obj["pos"][1] = 0
                obj["vel"][1] *= -0.8 # bounce
        
        print(f"Step {s}: {world_state}")

if __name__ == "__main__":
    init_world()
    run_sim(5)
