# AI-Driven Aero Design Accelerator (3D Phase)

An interactive 3D WebGL application designed to accelerate aerodynamic shape prototyping and evaluation. This platform enables side-by-side performance comparison between a high-fidelity **3D CFD (Computational Fluid Dynamics) Solver** and an **Instant SciML (Scientific Machine Learning) Neural Surrogate**.

---

## Developer Contact Information
*   **Developer**: Mohibul Hoque
*   **Email**: [hokworks@gmail.com](mailto:hokworks@gmail.com)
*   **LinkedIn**: [linkedin.com/in/speedymohibul](https://linkedin.com/in/speedymohibul)

---

## Project Overview

Evaluating aerodynamic geometries (such as wings, airfoils, and heat sinks) traditionally requires solving partial differential equations numerically, which can be computationally intensive and slow. This project solves that bottleneck by providing:
1.  **High-Fidelity Numeric CFD**: Using a 3D Lattice Boltzmann Method (LBM) solver to compute real physical flows.
2.  **Instant Deep Learning Inference**: Using a 3D U-Net surrogate trained to predict full 3D velocity ($u, v, w$) and pressure ($p$) flow fields, along with aerodynamic drag force, in milliseconds.
3.  **WebGL Visualizer Dashboard**: A gorgeous, real-time 3D dashboard displaying streamlines, heatmaps, and voxel obstacles.

---

## The Neural Network & Rationale Behind the Surrogate Model

### Why Choose a Surrogate Model?
Traditional CFD numerical methods (like our D3Q19 LBM solver) are mathematically rigorous but require thousands of iterative steps to solve the Navier-Stokes equations and reach convergence. In 3D domains, this becomes extremely slow—running the solver on a standard CPU takes approximately **23 seconds** per simulation. 

This delay makes real-time interactive design loops impossible. The **Neural Surrogate Model** acts as an emulator of the numerical solver. Once trained offline, the surrogate can predict flow patterns in **milliseconds** (and around 2.7 seconds on a single CPU thread), enabling immediate, real-time feedback as the user draws or modifies geometries.

### Neural Network Architecture: 3D U-Net
The surrogate uses a **3D U-Net (`UNet3D`)** architecture. This specific model was chosen for the following reasons:

1.  **Volume-to-Volume Translation**: Since both the input (voxel geometry) and the output (fluid velocity & pressure) reside on the same uniform `32x32x120` spatial grid, the problem is framed as a volumetric translation task.
2.  **Skip Connections**: The key strength of a U-Net is its skip connections, which link corresponding levels of the downsampling (encoder) and upsampling (decoder) paths. This allows the network to bypass bottleneck layers and directly preserve fine-grained spatial information, which is critical for mapping sharp velocity gradients and boundary layers right at the obstacle's surface.
3.  **Context and Fine Detail**: The encoder path reduces spatial dimensions to capture global context (such as the overall shape orientation and Reynolds number), while the decoder path recovers the local details (like wake turbulence and flow separation behind the object).

#### Model Input / Output Specifications:

*   **Inputs (Shape: `[1, 2, 32, 32, 120]`)**:
    *   **Channel 0**: The 3D binary voxel mask of the design shape ($1$ representing solid, $0$ representing fluid).
    *   **Channel 1**: The normalized Reynolds number ($Re / 150.0$) broadcasted/tiled across the entire 3D grid.
*   **Outputs (Shape: `[1, 4, 32, 32, 120]`)**:
    *   **Channel 0**: $u$ velocity component (along flow direction).
    *   **Channel 1**: $v$ velocity component (vertical direction).
    *   **Channel 2**: $w$ velocity component (spanwise direction).
    *   **Channel 3**: $p$ gauge pressure field.

---

## Getting Started Locally

### Prerequisites
*   Python 3.10 or 3.11
*   Node.js (for syntax validation tools, optional)

### Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your-repository-url>
   cd Aero-3d-Neural-network
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The requirements file pulls a lightweight CPU build of PyTorch to ensure fast load and build times).*

### Running the Server
Start the Uvicorn web server locally:
```bash
uvicorn backend.server_3d:app --host 127.0.0.1 --port 8000
```
Open your browser and navigate to:
`http://127.0.0.1:8000/index_3d.html`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
