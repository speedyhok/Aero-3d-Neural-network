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

## Architecture & Core Features

### 1. 3D CFD Solver
*   **Method**: Lattice Boltzmann Method (LBM) using the **D3Q19** (3-dimensions, 19 lattice velocities) collision scheme.
*   **Outputs**: Computes physical velocity vectors and pressure fields, enforcing no-slip bounce-back conditions on solid boundary walls.
*   **Drag Force Estimation**: Calculates viscous drag force by integrating pressure differences across the obstacle surfaces.

### 2. 3D Neural Surrogate (SciML)
*   **Model**: 3D U-Net architecture.
*   **Physics Constraints**: Trained offline using a combination of supervised loss and physical conservation constraints (PINN approach).
*   **Performance**: Completes flow predictions on complex extruded profiles in milliseconds on CPU, bypassing numerical solver iteration time.

### 3. Interactive Web Interface
*   **2D Design Workspace**: Draw custom cross-sectional profiles directly on the design board (extruded automatically to 3D).
*   **Preset Geometries**: Fast-load standard aerodynamic presets: Sphere, Cylinder, NACA Airfoil Wing, or Double Fin Heat Sink.
*   **3D Streamline Viz**: Dynamic WebGL particle streams showing flow separation, velocity magnitude gradients, and pressure slices.

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

## Deployment to Render.com (Web Service)

This application is ready to be hosted as a **Web Service** on Render.

1.  Push the code to a GitHub repository.
2.  Create a **New Web Service** on Render connected to your GitHub repository.
3.  Use the following configuration details in the Render settings panel:

| Setting | Value |
| :--- | :--- |
| **Runtime** | `Python` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn backend.server_3d:app --host 0.0.0.0 --port $PORT` |

Once deployed, visit your Web Service URL appending `/index_3d.html`:
`https://<your-subdomain>.onrender.com/index_3d.html`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
