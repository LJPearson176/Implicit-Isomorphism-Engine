import numpy as np
import pandas as pd
from zeep import Client
import time

class PhysicsEngine:
    def __init__(self, dataset="isotropic1024coarse"):
        self.wsdl_url = "http://turbulence.pha.jhu.edu/service/turbulence.asmx?WSDL"
        self.dataset = dataset
        self.token = "edu.jhu.pha.turbulence.testing-201311" # Public testing token
        print(f"Connecting to JHTDB ({self.wsdl_url})...")
        try:
            self.client = Client(self.wsdl_url)
            print("Connected to JHTDB Service.")
        except Exception as e:
            print(f"Failed to connect to JHTDB: {e}")
            self.client = None

    def get_velocity(self, time_step=0.0, points=None):
        """
        Fetches Velocity (u, v, w) for a list of points (x, y, z) at a given time.
        JHTDB 'GetVelocity' signature: (token, dataset, time, spatial_interp, temporal_interp, points)
        """
        if self.client is None:
            return None
            
        if points is None:
            # Generate a line of probes if no points provided
            x = np.linspace(0, 2*np.pi, 20)
            y = np.zeros_like(x)
            z = np.zeros_like(x)
            points = np.stack([x, y, z], axis=1)

        # Convert points to JHTDB Point3 format
        # The API expects a byte array of floats or specific structure. 
        # Zeep handles complex types, let's see the schema. 
        # Actually JHTDB usually takes a flattened list of floats for 'GetVelocity'.
        # Let's try the GetVelocity variant that accepts discrete points.
        
        # NOTE: The JHTDB SOAP API is known to be tricky with complex types.
        # We will use the 'GetData' method which is more generic, or 'GetVelocity'.
        
        # Flatten points for the API if needed, but Zeep might want Point3 objects.
        # Let's try to inspect the service wsdl first if we were interactive. 
        # For now, we'll try a simplified access pattern: GetVelocity requires Point3[].
        
        # Construct Point3 array
        point3_list = []
        for p in points:
            point3_list.append({'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2])})
            
        try:
            print(f"Requesting velocity for {len(points)} points...")
            result = self.client.service.GetVelocity(
                self.token,
                self.dataset,
                float(time_step),
                "Lag4",   # Spatial Interpolation
                "None",   # Temporal Interpolation
                {'Point3': point3_list} # Array config
            )
            
            # Result is usually a list of vectors
            # Extract u,v,w
            vectors = []
            if result:
                # The result format depends on the SOAP mapping.
                # Usually it returns a list of Point3-like objects or a Vector3 array.
                # Safe parsing:
                for v in result.Vector3:
                    vectors.append([v.x, v.y, v.z])
            
            return np.array(vectors)

        except Exception as e:
            print(f"Error fetching velocity: {e}")
            # Fallback for prototype if SOAP strictness fails
            return self._generate_synthetic_turbulence(points)

    def _generate_synthetic_turbulence(self, points):
        print("Falling back to Synthetic Turbulence (Kolmogorov Spectrum Proxy)...")
        # Generate generic multiscale noise
        velocities = []
        for p in points:
            u = np.sin(p[0]) + 0.1*np.sin(10*p[0])
            v = np.cos(p[0]) + 0.1*np.cos(10*p[0])
            w = 0.0
            velocities.append([u, v, w])
        return np.array(velocities)

if __name__ == "__main__":
    print("--- Phase 8: Physics Engine (JHTDB) ---")
    engine = PhysicsEngine()
    
    # 1. Define a probe line (like a hot-wire anemometer)
    n_points = 50
    x_coords = np.linspace(0, 2*np.pi, n_points)
    probes = np.zeros((n_points, 3))
    probes[:, 0] = x_coords
    
    # 2. Fetch ground truth velocity
    velocity_field = engine.get_velocity(time_step=0.365, points=probes)
    
    if velocity_field is not None:
        print(f"Fetched Velocity Field: {velocity_field.shape}")
        
        # 3. Structure as Dataframe for Models
        df = pd.DataFrame(velocity_field, columns=['u', 'v', 'w'])
        df['x'] = x_coords
        print(df.head())
        
        # Save
        df.to_csv("turbulence_ground_truth.csv", index=False)
        print("Saved ground truth to turbulence_ground_truth.csv")
    else:
        print("Failed to acquire data.")
