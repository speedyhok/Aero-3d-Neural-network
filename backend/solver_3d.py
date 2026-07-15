import numpy as np
import torch

class LBMSolver3D:
    def __init__(self, nx=120, ny=32, nz=32, Re=100.0, u_inflow=0.04, device=None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Re = Re
        self.u_inflow = u_inflow
        
        # Automatically choose GPU if available, else CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Characteristic length D (typical diameter of obstacle in lattice units)
        self.D = ny / 5.0
        
        # Viscosity based on Reynolds number: nu = u_inflow * D / Re
        self.nu = self.u_inflow * self.D / self.Re
        
        # Relaxation time tau = 3 * nu + 0.5
        self.tau = max(3.0 * self.nu + 0.5, 0.515)
        self.omega = 1.0 / self.tau
        
        # D3Q19 velocity vectors (c_ix, c_iy, c_iz) on device
        self.v = torch.tensor([
            [ 0,  0,  0],  # 0: rest
            
            [ 1,  0,  0],  # 1: right (+x)
            [-1,  0,  0],  # 2: left (-x)
            [ 0,  1,  0],  # 3: up (+y)
            [ 0, -1,  0],  # 4: down (-y)
            [ 0,  0,  1],  # 5: out (+z)
            [ 0,  0, -1],  # 6: in (-z)
            
            [ 1,  1,  0],  # 7
            [-1, -1,  0],  # 8
            [ 1, -1,  0],  # 9
            [-1,  1,  0],  # 10
            
            [ 1,  0,  1],  # 11
            [-1,  0, -1],  # 12
            [ 1,  0, -1],  # 13
            [-1,  0,  1],  # 14
            
            [ 0,  1,  1],  # 15
            [ 0, -1, -1],  # 16
            [ 0,  1, -1],  # 17
            [ 0, -1,  1]   # 18
        ], dtype=torch.int32, device=self.device)
        
        # Lattice weights for D3Q19 on device
        self.w = torch.tensor([
            12/36,                                                     # rest
            2/36, 2/36, 2/36, 2/36, 2/36, 2/36,                         # coordinate axes
            1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36  # plane diagonals
        ], dtype=torch.float32, device=self.device)
        
        # Opposite directions (for bounce-back boundary conditions) on device
        self.opposite = torch.tensor([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17], dtype=torch.long, device=self.device)
        
        # Initialize obstacle mask (0: fluid, 1: solid)
        self.mask = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.bool, device=self.device)
        
        # Density and velocity fields
        self.rho = torch.ones((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.u = torch.ones((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device) * self.u_inflow
        self.v_vel = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        self.w_vel = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        
        # Initialize distributions to equilibrium
        self.f = self.equilibrium(self.rho, self.u, self.v_vel, self.w_vel)
        
    def equilibrium(self, rho, u, v_vel, w_vel):
        feq = torch.zeros((19,) + rho.shape, dtype=torch.float32, device=self.device)
        usq = u**2 + v_vel**2 + w_vel**2
        
        for i in range(19):
            vu = self.v[i, 0] * u + self.v[i, 1] * v_vel + self.v[i, 2] * w_vel
            feq[i] = self.w[i] * rho * (1.0 + 3.0 * vu + 4.5 * vu**2 - 1.5 * usq)
            
        return feq

    def set_obstacle(self, mask):
        """Set the obstacle mask (boolean matrix of shape (nz, ny, nx))"""
        if isinstance(mask, np.ndarray):
            mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(self.device).bool()
        assert mask.shape == (self.nz, self.ny, self.nx), f"Mask shape {mask.shape} must match solver shape {(self.nz, self.ny, self.nx)}"
        self.mask = mask
        
    def step(self):
        # 1. Collision step
        feq = self.equilibrium(self.rho, self.u, self.v_vel, self.w_vel)
        self.f += self.omega * (feq - self.f)
        
        # Save pre-streaming state
        f_before_stream = self.f.clone()
        
        # 2. Streaming step (using fast torch.roll)
        for i in range(19):
            shift_z = int(self.v[i, 2])
            shift_y = int(self.v[i, 1])
            shift_x = int(self.v[i, 0])
            self.f[i] = torch.roll(self.f[i], shifts=(shift_z, shift_y, shift_x), dims=(0, 1, 2))
            
        # 3. Boundary Conditions
        
        # Left boundary: Inflow (constant velocity, density extrapolated)
        u_in = torch.ones((self.nz, self.ny), dtype=torch.float32, device=self.device) * self.u_inflow
        v_in = torch.zeros((self.nz, self.ny), dtype=torch.float32, device=self.device)
        w_in = torch.zeros((self.nz, self.ny), dtype=torch.float32, device=self.device)
        rho_in = self.rho[:, :, 1]
        feq_in = self.equilibrium(rho_in, u_in, v_in, w_in)
        for i in range(19):
            self.f[i, :, :, 0] = feq_in[i, :, :]
            
        # Right boundary: Outflow (zero-gradient)
        for i in range(19):
            self.f[i, :, :, -1] = self.f[i, :, :, -2]
            
        # Wall bounce-back (no-slip) for y boundaries (top/bottom)
        for i in range(19):
            if self.v[i, 1] < 0: # moving towards top wall (y=0)
                self.f[self.opposite[i], :, 0, :] = f_before_stream[i, :, 0, :]
            if self.v[i, 1] > 0: # moving towards bottom wall (y=ny-1)
                self.f[self.opposite[i], :, -1, :] = f_before_stream[i, :, -1, :]
                
        # Wall bounce-back (no-slip) for z boundaries (front/back)
        for i in range(19):
            if self.v[i, 2] < 0: # moving towards front wall (z=0)
                self.f[self.opposite[i], 0, :, :] = f_before_stream[i, 0, :, :]
            if self.v[i, 2] > 0: # moving towards back wall (z=nz-1)
                self.f[self.opposite[i], -1, :, :] = f_before_stream[i, -1, :, :]
                
        # 4. Obstacle Bounce-Back (No-Slip Link-wise Half-way)
        for i in range(19):
            shift_z = -int(self.v[i, 2])
            shift_y = -int(self.v[i, 1])
            shift_x = -int(self.v[i, 0])
            neighbor_is_solid = torch.roll(self.mask, shifts=(shift_z, shift_y, shift_x), dims=(0, 1, 2))
            bounce_mask = (~self.mask) & neighbor_is_solid
            self.f[self.opposite[i], bounce_mask] = f_before_stream[i, bounce_mask]
            
        # 5. Recompute macroscopic quantities (density, velocity)
        self.rho = torch.sum(self.f, dim=0)
        self.rho = torch.clamp(self.rho, min=1e-6)
        
        ux = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        uy = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        uz = torch.zeros((self.nz, self.ny, self.nx), dtype=torch.float32, device=self.device)
        for i in range(19):
            ux += self.f[i] * self.v[i, 0]
            uy += self.f[i] * self.v[i, 1]
            uz += self.f[i] * self.v[i, 2]
            
        self.u = ux / self.rho
        self.v_vel = uy / self.rho
        self.w_vel = uz / self.rho
        
        # Enforce zero velocity inside obstacles
        self.u[self.mask] = 0.0
        self.v_vel[self.mask] = 0.0
        self.w_vel[self.mask] = 0.0
        
    def run(self, max_steps=5000, tolerance=1e-5, callback=None):
        """Runs simulation until convergence or max_steps is reached."""
        u_old = self.u.clone()
        v_old = self.v_vel.clone()
        w_old = self.w_vel.clone()
        
        for step in range(max_steps):
            self.step()
            
            # Check convergence every 100 steps
            if step > 0 and step % 100 == 0:
                change = torch.mean(torch.sqrt((self.u - u_old)**2 + (self.v_vel - v_old)**2 + (self.w_vel - w_old)**2)).item()
                u_old = self.u.clone()
                v_old = self.v_vel.clone()
                w_old = self.w_vel.clone()
                
                if callback:
                    callback(step, change)
                    
                if change < tolerance:
                    return step
                    
        return max_steps
        
    def get_fields(self):
        """Returns physical quantities. In LBM, pressure p = rho * cs^2 where cs^2 = 1/3"""
        pressure = (self.rho - 1.0) / 3.0 # Gauge pressure
        return (self.u.cpu().numpy(),
                self.v_vel.cpu().numpy(),
                self.w_vel.cpu().numpy(),
                pressure.cpu().numpy())
