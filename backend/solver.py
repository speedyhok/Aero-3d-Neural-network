# ==========================================
# Developer: Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# ==========================================

import numpy as np

class LBMSolver:
    def __init__(self, nx=120, ny=40, Re=100.0, u_inflow=0.04):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.u_inflow = u_inflow
        
        # Characteristic length D (typical diameter of obstacle in lattice units)
        self.D = ny / 5.0 # Let's say diameter is 8 units for a grid height of 40
        
        # Viscosity based on Reynolds number: nu = u_inflow * D / Re
        self.nu = self.u_inflow * self.D / self.Re
        
        # Relaxation time tau = 3 * nu + 0.5
        # Constrain tau to be >= 0.515 to prevent numerical divergence
        self.tau = max(3.0 * self.nu + 0.5, 0.515)
        self.omega = 1.0 / self.tau
        
        # D2Q9 velocities
        self.v = np.array([
            [0,  0],  # 0: rest
            [1,  0],  # 1: right
            [0,  1],  # 2: up
            [-1, 0],  # 3: left
            [0, -1],  # 4: down
            [1,  1],  # 5: up-right
            [-1, 1],  # 6: up-left
            [-1,-1],  # 7: down-left
            [1, -1]   # 8: down-right
        ])
        
        # Weights
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        # Opposite directions (for bounce-back)
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Initialize obstacle mask (0: fluid, 1: solid)
        self.mask = np.zeros((self.ny, self.nx), dtype=bool)
        
        # Density and velocity fields
        self.rho = np.ones((self.ny, self.nx))
        self.u = np.ones((self.ny, self.nx)) * self.u_inflow
        self.v_vel = np.zeros((self.ny, self.nx))
        
        # Initialize distributions to equilibrium
        self.f = self.equilibrium(self.rho, self.u, self.v_vel)
        
    def equilibrium(self, rho, u, v_vel):
        is_1d = (len(rho.shape) == 1)
        if is_1d:
            rho = rho[:, np.newaxis]
            u = u[:, np.newaxis]
            v_vel = v_vel[:, np.newaxis]
            
        feq = np.zeros((9, rho.shape[0], rho.shape[1]))
        usq = u**2 + v_vel**2
        
        for i in range(9):
            vu = self.v[i, 0] * u + self.v[i, 1] * v_vel
            feq[i] = self.w[i] * rho * (1.0 + 3.0 * vu + 4.5 * vu**2 - 1.5 * usq)
            
        if is_1d:
            return feq.squeeze(axis=2)
        return feq

        
    def set_obstacle(self, mask):
        """Set the obstacle mask (boolean matrix of shape (ny, nx))"""
        assert mask.shape == (self.ny, self.nx), "Mask shape must match solver shape"
        self.mask = mask.astype(bool)
        
    def step(self):
        # 1. Collision step
        feq = self.equilibrium(self.rho, self.u, self.v_vel)
        self.f += self.omega * (feq - self.f)
        
        # Save pre-streaming state for bounce-back boundary conditions
        f_before_stream = self.f.copy()
        
        # 2. Streaming step
        for i in range(9):
            self.f[i] = np.roll(self.f[i], shift=(self.v[i, 1], self.v[i, 0]), axis=(0, 1))
            
        # 3. Boundary Conditions
        
        # Left boundary: Inflow (constant velocity, pressure extrapolated)
        # We prescribe u = u_inflow, v_vel = 0. We compute density from interior.
        # Simple equilibrium injection for stability:
        u_in = np.ones(self.ny) * self.u_inflow
        v_in = np.zeros(self.ny)
        # Extrapolate density from x=1
        rho_in = self.rho[:, 1]
        feq_in = self.equilibrium(rho_in, u_in, v_in)
        for i in range(9):
            self.f[i, :, 0] = feq_in[i, :]
            
        # Right boundary: Outflow (zero-gradient for all distributions)
        for i in range(9):
            self.f[i, :, -1] = self.f[i, :, -2]
            
        # Top and Bottom boundaries: Wall bounce-back (no-slip)
        for i in range(9):
            if self.v[i, 1] < 0: # moving towards top wall (y=0)
                self.f[self.opposite[i], 0, :] = f_before_stream[i, 0, :]
            if self.v[i, 1] > 0: # moving towards bottom wall (y=ny-1)
                self.f[self.opposite[i], -1, :] = f_before_stream[i, -1, :]

        # 4. Obstacle Bounce-Back (No-Slip Link-wise Half-way)
        for i in range(9):
            shift_y, shift_x = self.v[i, 1], self.v[i, 0]
            neighbor_is_solid = np.roll(self.mask, shift=(-shift_y, -shift_x), axis=(0, 1))
            bounce_mask = (~self.mask) & neighbor_is_solid
            self.f[self.opposite[i], bounce_mask] = f_before_stream[i, bounce_mask]
            
        # 5. Recompute macroscopic quantities (density, velocity)
        self.rho = np.sum(self.f, axis=0)
        
        # Prevent division by zero
        self.rho = np.maximum(self.rho, 1e-6)
        
        ux = np.zeros((self.ny, self.nx))
        uy = np.zeros((self.ny, self.nx))
        for i in range(9):
            ux += self.f[i] * self.v[i, 0]
            uy += self.f[i] * self.v[i, 1]
            
        self.u = ux / self.rho
        self.v_vel = uy / self.rho
        
        # Enforce zero velocity inside obstacles
        self.u[self.mask] = 0.0
        self.v_vel[self.mask] = 0.0
        
    def run(self, max_steps=5000, tolerance=1e-5, callback=None):
        """Runs simulation until convergence or max_steps is reached."""
        u_old = self.u.copy()
        v_old = self.v_vel.copy()
        
        for step in range(max_steps):
            self.step()
            
            # Check convergence every 100 steps
            if step > 0 and step % 100 == 0:
                change = np.mean(np.sqrt((self.u - u_old)**2 + (self.v_vel - v_old)**2))
                u_old = self.u.copy()
                v_old = self.v_vel.copy()
                
                if callback:
                    callback(step, change)
                    
                if change < tolerance:
                    return step
                    
        return max_steps
        
    def get_fields(self):
        """Returns physical quantities. In LBM, pressure p = rho * cs^2 where cs^2 = 1/3"""
        pressure = (self.rho - 1.0) / 3.0 # Gauge pressure
        return self.u.copy(), self.v_vel.copy(), pressure.copy()
