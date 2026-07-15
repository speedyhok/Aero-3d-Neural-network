# ==========================================
# Developer: Mohibul Hoque
# Email: hokworks@gmail.com
# LinkedIn: linkedin.com/in/speedymohibul
# ==========================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=4):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = DoubleConv3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = DoubleConv3D(32, 64)
        self.pool3 = nn.MaxPool3d(2)
        
        self.conv4 = DoubleConv3D(64, 128)
        
        # Decoder
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.conv5 = DoubleConv3D(128, 64)
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.conv6 = DoubleConv3D(64, 32)
        
        self.up3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.conv7 = DoubleConv3D(32, 16)
        
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        
        # Decoder
        u1 = self.up1(c4)
        # Handle shape alignment in case of slightly odd grid shapes
        if u1.shape != c3.shape:
            u1 = F.interpolate(u1, size=c3.shape[2:], mode='trilinear', align_corners=False)
        m1 = torch.cat([u1, c3], dim=1)
        c5 = self.conv5(m1)
        
        u2 = self.up2(c5)
        if u2.shape != c2.shape:
            u2 = F.interpolate(u2, size=c2.shape[2:], mode='trilinear', align_corners=False)
        m2 = torch.cat([u2, c2], dim=1)
        c6 = self.conv6(m2)
        
        u3 = self.up3(c6)
        if u3.shape != c1.shape:
            u3 = F.interpolate(u3, size=c1.shape[2:], mode='trilinear', align_corners=False)
        m3 = torch.cat([u3, c1], dim=1)
        c7 = self.conv7(m3)
        
        out = self.out_conv(c7)
        return out

# ----------------------------------------------------
# 3D Physics Loss Computation Functions
# ----------------------------------------------------
def get_derivatives_3d(field, dx=1.0):
    """Computes spatial derivatives using central finite differences in 3D (interior only)."""
    # field has shape (batch, depth_nz, height_ny, width_nx)
    # Slice other dimensions to 1:-1 to align shapes to (nz-2, ny-2, nx-2)
    df_dx = (field[:, 1:-1, 1:-1, 2:] - field[:, 1:-1, 1:-1, :-2]) / (2.0 * dx)
    df_dy = (field[:, 1:-1, 2:, 1:-1] - field[:, 1:-1, :-2, 1:-1]) / (2.0 * dx)
    df_dz = (field[:, 2:, 1:-1, 1:-1] - field[:, :-2, 1:-1, 1:-1]) / (2.0 * dx)
    return df_dx, df_dy, df_dz

def get_laplacian_3d(field, dx=1.0):
    """Computes Laplacian using central finite differences in 3D (interior only)."""
    # Output shape: (batch, nz-2, ny-2, nx-2)
    d2f_dx2 = (field[:, 1:-1, 1:-1, 2:] - 2.0 * field[:, 1:-1, 1:-1, 1:-1] + field[:, 1:-1, 1:-1, :-2]) / (dx**2)
    d2f_dy2 = (field[:, 1:-1, 2:, 1:-1] - 2.0 * field[:, 1:-1, 1:-1, 1:-1] + field[:, 1:-1, :-2, 1:-1]) / (dx**2)
    d2f_dz2 = (field[:, 2:, 1:-1, 1:-1] - 2.0 * field[:, 1:-1, 1:-1, 1:-1] + field[:, :-2, 1:-1, 1:-1]) / (dx**2)
    return d2f_dx2 + d2f_dy2 + d2f_dz2

def compute_pinn_loss_3d(pred, mask, Re, dx=1.0):
    """Computes mass and momentum conservation PINN losses for 3D flow predictions (interior only).
    
    pred: tensor of shape (batch, 4, nz, ny, nx) -> channels: (u, v, w, p)
    mask: tensor of shape (batch, 1, nz, ny, nx) -> 1 inside obstacle, 0 in fluid
    Re: tensor of shape (batch,) -> Reynolds number per batch element
    """
    u = pred[:, 0]
    v = pred[:, 1]
    w = pred[:, 2]
    p = pred[:, 3]
    
    # Slice the fluid mask to the interior (shape: batch, nz-2, ny-2, nx-2)
    fluid_mask = 1.0 - mask.squeeze(1)
    fluid_mask_int = fluid_mask[:, 1:-1, 1:-1, 1:-1]
    
    # 1. Spatial derivatives (Shapes: batch, nz-2, ny-2, nx-2)
    du_dx, du_dy, du_dz = get_derivatives_3d(u, dx)
    dv_dx, dv_dy, dv_dz = get_derivatives_3d(v, dx)
    dw_dx, dw_dy, dw_dz = get_derivatives_3d(w, dx)
    dp_dx, dp_dy, dp_dz = get_derivatives_3d(p, dx)
    
    # 2. Laplacians (Shapes: batch, nz-2, ny-2, nx-2)
    lap_u = get_laplacian_3d(u, dx)
    lap_v = get_laplacian_3d(v, dx)
    lap_w = get_laplacian_3d(w, dx)
    
    # 3. Mass Conservation: divergence of velocity = 0 (interior only)
    divergence = du_dx + dv_dy + dw_dz
    loss_div = torch.mean((divergence * fluid_mask_int) ** 2)
    
    # 4. Momentum Conservation (Navier-Stokes equations in 3D - interior only)
    # Slice velocity fields to interior to match derivative shapes
    u_int = u[:, 1:-1, 1:-1, 1:-1]
    v_int = v[:, 1:-1, 1:-1, 1:-1]
    w_int = w[:, 1:-1, 1:-1, 1:-1]
    
    # Reshape Re to enable division over batch
    Re_expanded = Re.view(-1, 1, 1, 1)
    
    # Momentum X residual
    res_x = u_int * du_dx + v_int * du_dy + w_int * du_dz + dp_dx - (1.0 / Re_expanded) * lap_u
    # Momentum Y residual
    res_y = u_int * dv_dx + v_int * dv_dy + w_int * dv_dz + dp_dy - (1.0 / Re_expanded) * lap_v
    # Momentum Z residual
    res_z = u_int * dw_dx + v_int * dw_dy + w_int * dw_dz + dp_dz - (1.0 / Re_expanded) * lap_w
    
    loss_mom = torch.mean((res_x * fluid_mask_int) ** 2) + \
               torch.mean((res_y * fluid_mask_int) ** 2) + \
               torch.mean((res_z * fluid_mask_int) ** 2)
               
    # 5. Boundary Condition: No-Slip (velocity must be 0 inside obstacle mask - evaluated over full grid)
    loss_bound = torch.mean((u * mask.squeeze(1)) ** 2) + \
                 torch.mean((v * mask.squeeze(1)) ** 2) + \
                 torch.mean((w * mask.squeeze(1)) ** 2)
                 
    return loss_div, loss_mom, loss_bound

