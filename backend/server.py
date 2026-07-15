# ==========================================
# Developer: Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# ==========================================

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.solver import LBMSolver

app = FastAPI(title="AI-Driven Aero Design Accelerator API (Phase 1)")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Request and Response Models
# ----------------------------------------------------
class SimulationRequest(BaseModel):
    mask: List[List[int]]  # 2D list of shape (ny, nx)
    Re: float
    steps: int = 2500

class SimulationResponse(BaseModel):
    u: List[List[float]]
    v: List[List[float]]
    p: List[List[float]]
    steps_taken: int

# ----------------------------------------------------
# REST API Endpoints
# ----------------------------------------------------
@app.post("/api/solve", response_model=SimulationResponse)
def solve_cfd(req: SimulationRequest):
    mask_arr = np.array(req.mask, dtype=bool)
    ny, nx = mask_arr.shape
    
    if ny != 40 or nx != 120:
        raise HTTPException(status_code=400, detail="Grid dimensions must be exactly 120x40.")
        
    solver = LBMSolver(nx=nx, ny=ny, Re=req.Re, u_inflow=0.04)
    solver.set_obstacle(mask_arr)
    
    steps_taken = solver.run(max_steps=req.steps, tolerance=1e-5)
    u_field, v_field, p_field = solver.get_fields()
    
    return SimulationResponse(
        u=u_field.tolist(),
        v=v_field.tolist(),
        p=p_field.tolist(),
        steps_taken=steps_taken
    )

# Serve Frontend static files (HTML, CSS, JS)
# We place this at the very bottom so it doesn't hijack API routes
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    print(f"Mounted static frontend files from {frontend_dir}")
else:
    print(f"WARNING: Frontend folder not found at '{frontend_dir}'. API endpoints are active, but dashboard page is not served.")
