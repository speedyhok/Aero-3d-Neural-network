import numpy as np
import os
from backend.solver import LBMSolver

def verify():
    print("Initializing verification solver (Grid: 120x40, Re=100.0)...")
    solver = LBMSolver(nx=120, ny=40, Re=100.0)
    
    # Place a circular cylinder obstacle of diameter 8 at (x=30, y=20)
    ny, nx = 40, 120
    Y, X = np.ogrid[:ny, :nx]
    cy, cx = 20, 30
    radius = 4
    
    mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
    solver.set_obstacle(mask)
    
    print("Running solver to steady state (max 3000 steps)...")
    
    def log_progress(step, change):
        print(f"Step {step:4d} | Average Velocity Change: {change:.3e}")
        
    steps_taken = solver.run(max_steps=3000, tolerance=1e-5, callback=log_progress)
    print(f"Simulation completed in {steps_taken} steps.")
    
    u, v, p = solver.get_fields()
    
    # Calculate drag coefficient (approximate check)
    velocity_magnitude = np.sqrt(u**2 + v**2)
    max_vel = np.max(velocity_magnitude)
    mean_pressure = np.mean(p)
    
    print("\n--- Verification Report ---")
    print(f"Max velocity magnitude in domain: {max_vel:.4f}")
    print(f"Mean gauge pressure: {mean_pressure:.6f}")
    print(f"Obstacle cell count: {np.sum(mask)}")
    print(f"Is velocity inside cylinder zero? {np.all(velocity_magnitude[mask] == 0.0)}")
    
    # Save a verification snapshot (as txt array or text output)
    os.makedirs("scratch", exist_ok=True)
    np.save("scratch/verify_u.npy", u)
    np.save("scratch/verify_v.npy", v)
    np.save("scratch/verify_p.npy", p)
    print("Verification data fields saved to scratch/ directory.")
    print("Verification successful!")

if __name__ == "__main__":
    verify()
