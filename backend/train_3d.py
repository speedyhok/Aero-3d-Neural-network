# ==========================================
# Developer: Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# ==========================================

import os
import argparse
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from backend.solver_3d import LBMSolver3D
from backend.model_3d import UNet3D, compute_pinn_loss_3d

# ----------------------------------------------------
# 3D Shape Generator Functions
# ----------------------------------------------------
def generate_naca_profile_2d(chord, AoA_deg, camber, camberPos, thickness, cx, cy, nx, ny):
    """Generates a 2D NACA airfoil profile on a grid."""
    AoA = AoA_deg * np.pi / 180.0
    nPoints = 80
    xChord = np.linspace(0, 1, nPoints + 1)
    
    yc = np.zeros(nPoints + 1)
    dyc_dx = np.zeros(nPoints + 1)
    m = camber
    p = camberPos
    
    for i, x in enumerate(xChord):
        if x < p:
            if p > 0:
                yc[i] = (m / (p**2)) * (2 * p * x - x**2)
                dyc_dx[i] = (2 * m / (p**2)) * (p - x)
        else:
            if p < 1:
                yc[i] = (m / ((1 - p)**2)) * ((1 - 2 * p) + 2 * p * x - x**2)
                dyc_dx[i] = (2 * m / ((1 - p)**2)) * (p - x)
                
    xu = np.zeros(nPoints + 1)
    yu = np.zeros(nPoints + 1)
    xl = np.zeros(nPoints + 1)
    yl = np.zeros(nPoints + 1)
    t = thickness
    
    for i, x in enumerate(xChord):
        theta = np.arctan(dyc_dx[i])
        yt = 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
        xu[i] = x - yt * np.sin(theta)
        yu[i] = yc[i] + yt * np.cos(theta)
        xl[i] = x + np.sin(theta) * yt
        yl[i] = yc[i] - np.cos(theta) * yt
        
    poly_points = []
    cosA, sinA = np.cos(AoA), np.sin(AoA)
    for i in range(nPoints + 1):
        xr = xu[i] * chord * cosA - yu[i] * chord * sinA + cx
        yr = xu[i] * chord * sinA + yu[i] * chord * cosA + cy
        poly_points.append((xr, yr))
    for i in range(nPoints, -1, -1):
        xr = xl[i] * chord * cosA - yl[i] * chord * sinA + cx
        yr = xl[i] * chord * sinA + yl[i] * chord * cosA + cy
        poly_points.append((xr, yr))
        
    # Ray casting to fill 2D mask
    mask_2d = np.zeros((ny, nx), dtype=bool)
    for y in range(ny):
        for x in range(nx):
            # ray cast check
            inside = False
            for i in range(len(poly_points)):
                j = (i - 1) % len(poly_points)
                xi, yi = poly_points[i]
                xj, yj = poly_points[j]
                intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-8) + xi)
                if intersect:
                    inside = not inside
            if inside:
                mask_2d[y, x] = True
    return mask_2d

def generate_random_obstacle_3d(nz=32, ny=32, nx=120):
    """Creates a random 3D obstacle mask."""
    mask = np.zeros((nz, ny, nx), dtype=bool)
    shape_type = np.random.choice(["sphere", "cylinder", "wing", "box"])
    
    cx = np.random.randint(25, 35)
    cy = np.random.randint(12, 20)
    cz = np.random.randint(12, 20)
    
    Z, Y, X = np.ogrid[:nz, :ny, :nx]
    
    if shape_type == "sphere":
        r = np.random.uniform(3.0, 5.5)
        mask[:, :, :] = (X - cx)**2 + (Y - cy)**2 + (Z - cz)**2 <= r**2
        
    elif shape_type == "cylinder":
        r = np.random.uniform(2.5, 4.5)
        # Random axis alignment: X-axis (0), Y-axis (1), Z-axis (2)
        axis = np.random.randint(0, 3)
        if axis == 0:
            mask[:, :, :] = (Y - cy)**2 + (Z - cz)**2 <= r**2
            # Limit length along x
            mask[:, :, :20] = False
            mask[:, :, 45:] = False
        elif axis == 1:
            mask[:, :, :] = (X - cx)**2 + (Z - cz)**2 <= r**2
        else:
            mask[:, :, :] = (X - cx)**2 + (Y - cy)**2 <= r**2
            
    elif shape_type == "box":
        dx = np.random.randint(4, 9)
        dy = np.random.randint(4, 9)
        dz = np.random.randint(4, 9)
        mask[:, :, :] = (np.abs(X - cx) <= dx) & (np.abs(Y - cy) <= dy) & (np.abs(Z - cz) <= dz)
        
    elif shape_type == "wing":
        # Extruded NACA airfoil with slight tilt
        chord = np.random.uniform(15.0, 25.0)
        AoA = np.random.uniform(-4.0, 12.0)
        camber = np.random.uniform(0.0, 0.06)
        camber_pos = np.random.uniform(0.2, 0.5)
        thickness = np.random.uniform(0.08, 0.16)
        
        # Draw 2D profile
        mask_2d = generate_naca_profile_2d(chord, AoA, camber, camber_pos, thickness, cx, cy, nx, ny)
        
        # Extrude along Z-axis (axis 0)
        for z in range(nz):
            mask[z, :, :] = mask_2d
            
    # Clear flow margins (inflow, outflow, walls)
    mask[:, :, :5] = False
    mask[:, :, -5:] = False
    mask[:, 0, :] = False
    mask[:, -1, :] = False
    mask[0, :, :] = False
    mask[-1, :, :] = False
    
    return mask

# ----------------------------------------------------
# Data Generation Pipeline
# ----------------------------------------------------
def generate_single_case(case_idx, output_dir, nx=120, ny=32, nz=32, max_steps=1500):
    """Simulates a single case and saves the fields."""
    np.random.seed(case_idx)
    try:
        mask = generate_random_obstacle_3d(nz, ny, nx)
        Re = np.random.uniform(10.0, 150.0)
        
        solver = LBMSolver3D(nx=nx, ny=ny, nz=nz, Re=Re, u_inflow=0.04)
        solver.set_obstacle(mask)
        
        # Quick convergence parameters for training generation
        steps_run = solver.run(max_steps=max_steps, tolerance=1e-4)
        u, v, w, p = solver.get_fields()
        
        filepath = os.path.join(output_dir, f"case_{case_idx:04d}.npz")
        np.savez_compressed(
            filepath,
            mask=mask.astype(np.uint8),
            u=u.astype(np.float32),
            v=v.astype(np.float32),
            w=w.astype(np.float32),
            p=p.astype(np.float32),
            Re=float(Re)
        )
        print(f"Saved case {case_idx:04d} | Re: {Re:.1f} | Steps: {steps_run}")
    except Exception as e:
        print(f"Error generating case {case_idx}: {e}")

# ----------------------------------------------------
# PyTorch Dataset
# ----------------------------------------------------
class CFD3DDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        mask = torch.tensor(data['mask'], dtype=torch.float32).unsqueeze(0) # (1, nz, ny, nx)
        Re = torch.tensor(data['Re'], dtype=torch.float32)
        
        u = torch.tensor(data['u'], dtype=torch.float32)
        v = torch.tensor(data['v'], dtype=torch.float32)
        w = torch.tensor(data['w'], dtype=torch.float32)
        p = torch.tensor(data['p'], dtype=torch.float32)
        
        target = torch.stack([u, v, w, p], dim=0) # (4, nz, ny, nx)
        return mask, Re, target

# ----------------------------------------------------
# Training Runner
# ----------------------------------------------------
def train_model(data_dir, epochs=10, batch_size=2, lr=1e-3):
    print("Initializing 3D Surrogate Model Training Loop...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = CFD3DDataset(data_dir)
    if len(dataset) == 0:
        print("Error: No training dataset files found. Run data generation first using --generate.")
        return
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = UNet3D(in_channels=2, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_loss = float('inf')
    save_path = "backend/surrogate_model_3d.pth"
    os.makedirs("backend", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for mask, Re, target in train_loader:
            mask = mask.to(device)
            Re = Re.to(device)
            target = target.to(device)
            
            # Setup input stack: channel 0 = mask, channel 1 = tiled Re
            Re_tiled = Re.view(-1, 1, 1, 1, 1).expand(-1, 1, mask.shape[2], mask.shape[3], mask.shape[4])
            inputs = torch.cat([mask, Re_tiled / 150.0], dim=1) # normalize Re
            
            optimizer.zero_grad()
            pred = model(inputs)
            
            # 1. Supervised MSE Data Loss
            loss_mse = F_mse = nn.MSELoss()(pred, target)
            
            # 2. Physics-Informed (PINN) Loss
            loss_div, loss_mom, loss_bound = compute_pinn_loss_3d(pred, mask, Re)
            loss_pinn = 0.1 * loss_div + 0.1 * loss_mom + 1.0 * loss_bound
            
            # Combined Loss
            loss_total = loss_mse + 0.01 * loss_pinn
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_total.item() * mask.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation Pass
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mask, Re, target in val_loader:
                mask = mask.to(device)
                Re = Re.to(device)
                target = target.to(device)
                
                Re_tiled = Re.view(-1, 1, 1, 1, 1).expand(-1, 1, mask.shape[2], mask.shape[3], mask.shape[4])
                inputs = torch.cat([mask, Re_tiled / 150.0], dim=1)
                
                pred = model(inputs)
                loss_mse = nn.MSELoss()(pred, target)
                loss_div, loss_mom, loss_bound = compute_pinn_loss_3d(pred, mask, Re)
                loss_pinn = 0.1 * loss_div + 0.1 * loss_mom + 1.0 * loss_bound
                loss_total = loss_mse + 0.01 * loss_pinn
                val_loss += loss_total.item() * mask.size(0)
                
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model checkpoint to {save_path}")
            
    print("Training process finished.")

# ----------------------------------------------------
# Main Execution Entry Point
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D SciML Training & Data Pipeline")
    parser.add_argument("--generate", action="store_true", help="Generate the 3D CFD dataset")
    parser.add_argument("--num_cases", type=int, default=100, help="Number of 3D flow cases to generate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=1500, help="Max solver steps per simulation")
    args = parser.parse_args()
    
    data_dir = "data/cfd_dataset_3d"
    
    if args.generate:
        os.makedirs(data_dir, exist_ok=True)
        print(f"Generating 3D CFD dataset containing {args.num_cases} cases...")
        
        # Check if CUDA is available for solver acceleration
        if torch.cuda.is_available():
            print("CUDA GPU detected! Running sequential data generation on GPU...")
            for i in range(args.num_cases):
                generate_single_case(i, data_dir, max_steps=args.max_steps)
        else:
            print("No CUDA GPU detected. Running parallel data generation on CPU...")
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            print(f"Using {num_cores} parallel CPU cores...")
            import functools
            worker_func = functools.partial(generate_single_case, output_dir=data_dir, max_steps=args.max_steps)
            pool_args = list(range(args.num_cases))
            with multiprocessing.Pool(num_cores) as pool:
                pool.map(worker_func, pool_args)
            
        print("Data generation completed successfully!")
        
    else:
        train_model(data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
