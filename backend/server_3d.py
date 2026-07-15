import os
import time
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List

from backend.solver_3d import LBMSolver3D
from backend.model_3d import UNet3D

app = FastAPI(title="AI-Driven Aero Design Accelerator API (3D Phase)")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Global Model Loading
# ----------------------------------------------------
weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "surrogate_model_3d.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_loaded = False

try:
    if os.path.exists(weights_path):
        model = UNet3D(in_channels=2, out_channels=4).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        model_loaded = True
        print(f"3D Surrogate Model successfully loaded on {device} from '{weights_path}'")
    else:
        print(f"WARNING: 3D Surrogate model weights not found at '{weights_path}'. Predict route will be disabled.")
except Exception as e:
    print(f"Error initializing 3D surrogate model: {e}")

# ----------------------------------------------------
# Request and Response Schemas
# ----------------------------------------------------
class SimulationRequest3D(BaseModel):
    mask: List[List[List[int]]]  # 3D list of shape (nz, ny, nx) = (32, 32, 120)
    Re: float
    steps: int = 500


# Helper function to compute drag force from pressure field and obstacle mask
def calculate_drag_force_3d(p, mask):
    """Computes pressure drag force along the x-axis (flow direction) in 3D."""
    nz, ny, nx = mask.shape
    mask_float = mask.astype(float)
    
    # Left boundary (fluid cell at x-1 pushes right onto obstacle cell x)
    left_boundary = (mask_float == 1.0) & (np.roll(mask_float, shift=1, axis=2) == 0.0)
    # Right boundary (fluid cell at x+1 pushes left onto obstacle cell x)
    right_boundary = (mask_float == 1.0) & (np.roll(mask_float, shift=-1, axis=2) == 0.0)
    
    # Exclude boundary margins of the grid
    left_boundary[:, :, 0] = False
    right_boundary[:, :, -1] = False
    
    # Extract pressures in the adjacent fluid cells
    p_left = np.roll(p, shift=1, axis=2)[left_boundary]
    p_right = np.roll(p, shift=-1, axis=2)[right_boundary]
    
    fx = np.sum(p_left) - np.sum(p_right)
    return float(fx)

# ----------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------
@app.get("/api/model_status")
def model_status():
    return {
        "surrogate_loaded": model_loaded,
        "device": "cpu",
        "grid_dimensions": {"nz": 32, "ny": 32, "nx": 120}
    }

@app.post("/api/predict")
def predict_flow_3d(req: SimulationRequest3D):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="3D Surrogate model is not loaded on this server.")
        
    mask_arr = np.array(req.mask, dtype=np.float32)
    nz, ny, nx = mask_arr.shape
    
    if nz != 32 or ny != 32 or nx != 120:
        raise HTTPException(status_code=400, detail="Grid dimensions must be exactly 32x32x120 (nz x ny x nx).")
        
    t_start = time.time()
    
    # Prepare inputs for the PyTorch U-Net 3D model
    mask_tensor = torch.tensor(mask_arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    Re_val = torch.tensor([req.Re], dtype=torch.float32).to(device)
    
    # Re tiled to match input shape
    Re_tiled = Re_val.view(-1, 1, 1, 1, 1).expand(-1, 1, nz, ny, nx)
    inputs = torch.cat([mask_tensor, Re_tiled / 150.0], dim=1) # (1, 2, nz, ny, nx)
    
    with torch.inference_mode():
        pred = model(inputs) # (1, 4, nz, ny, nx)
        
    u = pred[0, 0].cpu().numpy()
    v = pred[0, 1].cpu().numpy()
    w = pred[0, 2].cpu().numpy()
    p = pred[0, 3].cpu().numpy()
    
    # Enforce exact physical zeros inside obstacle boundaries
    obstacle_mask = (mask_arr > 0.5)
    u[obstacle_mask] = 0.0
    v[obstacle_mask] = 0.0
    w[obstacle_mask] = 0.0
    
    drag = calculate_drag_force_3d(p, obstacle_mask)
    t_elapsed = time.time() - t_start
    
    # Flatten and pack u, v, w, p into a single binary float32 buffer (4 * 32 * 32 * 120 float32s)
    data_bytes = np.stack([u, v, w, p], axis=0).astype(np.float32).tobytes()
    
    headers = {
        "X-Drag": str(drag),
        "X-Execution-Time-Ms": str(int(t_elapsed * 1000))
    }
    return Response(content=data_bytes, media_type="application/octet-stream", headers=headers)

@app.post("/api/solve")
def solve_cfd_3d(req: SimulationRequest3D):
    mask_arr = np.array(req.mask, dtype=bool)
    nz, ny, nx = mask_arr.shape
    
    if nz != 32 or ny != 32 or nx != 120:
        raise HTTPException(status_code=400, detail="Grid dimensions must be exactly 32x32x120 (nz x ny x nx).")
        
    t_start = time.time()
    
    # Initialize 3D LBM Solver (force CPU for web requests)
    solver = LBMSolver3D(nx=nx, ny=ny, nz=nz, Re=req.Re, u_inflow=0.04, device="cpu")
    solver.set_obstacle(mask_arr)
    
    # Run solver for specified steps
    steps_taken = solver.run(max_steps=req.steps, tolerance=5e-5)
    u, v, w, p = solver.get_fields()
    
    drag = calculate_drag_force_3d(p, mask_arr)
    t_elapsed = time.time() - t_start
    
    # Flatten and pack u, v, w, p into a single binary float32 buffer (4 * 32 * 32 * 120 float32s)
    data_bytes = np.stack([u, v, w, p], axis=0).astype(np.float32).tobytes()
    
    headers = {
        "X-Drag": str(drag),
        "X-Execution-Time-Ms": str(int(t_elapsed * 1000)),
        "X-Steps-Taken": str(steps_taken)
    }
    return Response(content=data_bytes, media_type="application/octet-stream", headers=headers)




# Serve Frontend static files (HTML, CSS, JS)
# Placed at the bottom of the server file to prevent path conflicts
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

@app.get("/")
def read_index():
    return FileResponse(os.path.join(frontend_dir, "index_3d.html"))

if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"Serving frontend static files from: {frontend_dir}")
else:
    print(f"WARNING: Frontend folder not found at '{frontend_dir}'")
