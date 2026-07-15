// ----------------------------------------------------
// AERO-DECO 3D: WebGL Core App Logic
// ----------------------------------------------------

const NZ = 32;
const NY = 32;
const NX = 120;

// Application State
let activePreset = "sphere";
let brushMode = "draw"; // "draw" or "erase"
let brushSize = 2;
let reynoldsNumber = 100;
let renderMode = "particles"; // "particles", "slice", "voxels"
let sliceAxis = "z";
let sliceIndex = 16;

// 3D Grid State
let obstacleMask = Array.from({ length: NZ }, () =>
    Array.from({ length: NY }, () => new Uint8Array(NX))
);

// Comparative Viewports (CFD vs Neural Surrogate)
let viewports = {
    cfd: {
        containerId: "cfdViewport",
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        particles: null,
        voxelGroup: null,
        slicePlane: null,
        lineSegments: null,
        particleData: [],
        flowFields: { u: null, v: null, w: null, p: null, drag: null }
    },
    surrogate: {
        containerId: "surrogateViewport",
        scene: null,
        camera: null,
        renderer: null,
        controls: null,
        particles: null,
        voxelGroup: null,
        slicePlane: null,
        lineSegments: null,
        particleData: [],
        flowFields: { u: null, v: null, w: null, p: null, drag: null }
    }
};

// 2D Draw Canvas properties
const canvas2D = document.getElementById("designCanvas2D");
const ctx2D = canvas2D.getContext("2d");
let isDrawing = false;



// Speed colormap: Deep Violet (stagnation) -> Neon Cyan (normal) -> Neon Green (accelerated) -> Gold (max speed)
function getFlowColor(speed, maxVal = 0.09) {
    const t = Math.min(1.0, Math.max(0.0, speed / maxVal));
    
    let r, g, b;
    if (t < 0.3) {
        // Stagnation to normal: Violet (100, 0, 200) to Cyan (0, 200, 255)
        const ratio = t / 0.3;
        r = (1.0 - ratio) * 0.4;
        g = ratio * 0.8;
        b = 0.8 + ratio * 0.2;
    } else if (t < 0.7) {
        // Normal to accelerated: Cyan (0, 200, 255) to Green (0, 255, 100)
        const ratio = (t - 0.3) / 0.4;
        r = 0.0;
        g = 0.8 + ratio * 0.2;
        b = 1.0 - ratio * 0.6;
    } else {
        // Accelerated to max: Green (0, 255, 100) to Gold (255, 220, 0)
        const ratio = (t - 0.7) / 0.3;
        r = ratio;
        g = 1.0 - ratio * 0.14;
        b = 0.39 * (1.0 - ratio);
    }
    
    return { r, g, b };
}

// Interaction colormap: shift color towards obstacle's hot pink (#fe0879 = 1.0, 0.03, 0.47) when close to surface
function getInteractionColor(speed, proximity) {
    const base = getFlowColor(speed);
    if (proximity <= 0) return base;
    
    const heat = Math.pow(proximity, 1.2); // blend factor
    
    const r = base.r + (1.0 - base.r) * heat;
    const g = base.g * (1.0 - heat);
    const b = base.b + (0.47 - base.b) * heat;
    
    return { r, g, b };
}

// Precomputed distance field: obstacleDistField[z*NY*NX + y*NX + x] = proximity (0..1)
// Built once per obstacle change in buildObstacleDistanceField(), queried O(1) per particle
let obstacleDistField = null;

const INTERACT_RADIUS = 4.5; // voxels — glow starts this many voxels from surface

function buildObstacleDistanceField() {
    const size = NZ * NY * NX;
    obstacleDistField = new Float32Array(size); // default 0 = no proximity
    const R = Math.ceil(INTERACT_RADIUS);
    for (let z0 = 0; z0 < NZ; z0++) {
        for (let y0 = 0; y0 < NY; y0++) {
            for (let x0 = 0; x0 < NX; x0++) {
                if (obstacleMask[z0][y0][x0] === 1) continue; // inside solid
                // Find closest solid voxel within INTERACT_RADIUS
                let minDist2 = Infinity;
                for (let dz = -R; dz <= R; dz++) {
                    const z = z0 + dz; if (z < 0 || z >= NZ) continue;
                    for (let dy = -R; dy <= R; dy++) {
                        const y = y0 + dy; if (y < 0 || y >= NY) continue;
                        for (let dx = -R; dx <= R; dx++) {
                            const x = x0 + dx; if (x < 0 || x >= NX) continue;
                            if (obstacleMask[z][y][x] === 1) {
                                const d2 = dx*dx + dy*dy + dz*dz;
                                if (d2 < minDist2) minDist2 = d2;
                            }
                        }
                    }
                }
                if (minDist2 < Infinity) {
                    const dist = Math.sqrt(minDist2);
                    obstacleDistField[z0*NY*NX + y0*NX + x0] = Math.max(0, 1.0 - dist / INTERACT_RADIUS);
                }
            }
        }
    }
}

// O(1) proximity lookup — clamps coords and reads precomputed table
function getObstacleProximity(px, py, pz) {
    if (!obstacleDistField) return 0;
    const x = Math.min(NX-1, Math.max(0, Math.floor(px)));
    const y = Math.min(NY-1, Math.max(0, Math.floor(py)));
    const z = Math.min(NZ-1, Math.max(0, Math.floor(pz)));
    return obstacleDistField[z*NY*NX + y*NX + x];
}

// ----------------------------------------------------
// UI Initialization
// ----------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    init3DViewports();
    initPresets();
    init2DDrawCanvas();
    initControls();
    checkModelStatus();
    
    // Trigger initial preset render with sphere, then auto-predict
    loadPreset("sphere");
    initializeDefaultLaminarFlow();
    
    // Animation loop
    animate();
    
    // Auto-run surrogate after a short delay to let the scene fully init
    setTimeout(() => runSurrogatePredict(), 500);
});

// Sync camera movements across both viewports
function syncCameras(sourceControls, targetCamera, targetRenderer, targetScene) {
    targetCamera.position.copy(sourceControls.object.position);
    targetCamera.rotation.copy(sourceControls.object.rotation);
    targetCamera.zoom = sourceControls.object.zoom;
    targetCamera.lookAt(sourceControls.target);
}

function init3DViewports() {
    const keys = ["cfd", "surrogate"];
    keys.forEach((key) => {
        const vp = viewports[key];
        const container = document.getElementById(vp.containerId);
        const width = container.clientWidth;
        const height = container.clientHeight;
        
        // Scene
        vp.scene = new THREE.Scene();
        vp.scene.background = new THREE.Color(0x040508);
        
        // Camera
        vp.camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 1000);
        vp.camera.position.set(0, 45, 120);
        
        // Renderer
        vp.renderer = new THREE.WebGLRenderer({ antialias: true });
        vp.renderer.setSize(width, height);
        vp.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(vp.renderer.domElement);
        
        // Controls
        vp.controls = new THREE.OrbitControls(vp.camera, vp.renderer.domElement);
        vp.controls.enableDamping = true;
        vp.controls.dampingFactor = 0.05;
        vp.controls.maxPolarAngle = Math.PI / 2 + 0.1; // Limit under-floor rotation
        vp.controls.minDistance = 30;
        vp.controls.maxDistance = 300;
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        vp.scene.add(ambientLight);
        
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(20, 40, 20);
        vp.scene.add(dirLight);
        
        // Draw grid bounding box boundary lines
        const boxGeom = new THREE.BoxGeometry(NX, NY, NZ);
        const edges = new THREE.EdgesGeometry(boxGeom);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x1f2b38, linewidth: 1 });
        const boxFrame = new THREE.LineSegments(edges, lineMat);
        vp.scene.add(boxFrame);
        
        // Voxel Group
        vp.voxelGroup = new THREE.Group();
        vp.scene.add(vp.voxelGroup);
        
        // Slice Plane Mesh (initialized as hidden)
        const planeGeom = new THREE.PlaneGeometry(1, 1);
        const planeMat = new THREE.MeshBasicMaterial({ 
            side: THREE.DoubleSide, 
            transparent: true,
            opacity: 0.85
        });
        vp.slicePlane = new THREE.Mesh(planeGeom, planeMat);
        vp.slicePlane.visible = false;
        vp.scene.add(vp.slicePlane);
        
        // Flow Particle Streamlines Setup
        initParticles(key);
    });
    
    // Attach sync handlers
    viewports.cfd.controls.addEventListener("change", () => {
        syncCameras(viewports.cfd.controls, viewports.surrogate.camera, viewports.surrogate.renderer, viewports.surrogate.scene);
    });
    viewports.surrogate.controls.addEventListener("change", () => {
        syncCameras(viewports.surrogate.controls, viewports.cfd.camera, viewports.cfd.renderer, viewports.cfd.scene);
    });
    
    window.addEventListener("resize", () => {
        keys.forEach((key) => {
            const vp = viewports[key];
            const container = document.getElementById(vp.containerId);
            vp.camera.aspect = container.clientWidth / container.clientHeight;
            vp.camera.updateProjectionMatrix();
            vp.renderer.setSize(container.clientWidth, container.clientHeight);
        });
    });
}

function initParticles(key) {
    const vp = viewports[key];
    const particleCount = 7500;
    
    // Each particle has 3 segments = 6 vertices
    const positions = new Float32Array(particleCount * 6 * 3);
    const colors = new Float32Array(particleCount * 6 * 3);
    
    vp.particleData = [];
    
    for (let i = 0; i < particleCount; i++) {
        const px = 1.0 + Math.random() * 4.0;  // x = 1..5
        const py = 1.0 + Math.random() * (NY - 2.0);
        const pz = 1.0 + Math.random() * (NZ - 2.0);
        
        // Initial coordinate history
        const history = Array.from({ length: 4 }, () => ({ x: px, y: py, z: pz }));
        
        vp.particleData.push({
            x: px,
            y: py,
            z: pz,
            history: history,
            life: Math.random() * 450
        });
        
        const idx = i * 6;
        for (let j = 0; j < 6; j++) {
            positions[(idx + j) * 3]     = px - NX / 2;
            positions[(idx + j) * 3 + 1] = py - NY / 2;
            positions[(idx + j) * 3 + 2] = pz - NZ / 2;
            
            const col = getFlowColor(0.04);
            colors[(idx + j) * 3]     = col.r;
            colors[(idx + j) * 3 + 1] = col.g;
            colors[(idx + j) * 3 + 2] = col.b;
        }
    }
    
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    
    // Line material for aerodynamic streaks
    const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        blending: THREE.AdditiveBlending,
        linewidth: 1.5
    });
    
    vp.particles = new THREE.LineSegments(geometry, material);
    vp.scene.add(vp.particles);
}

// ----------------------------------------------------
// Flow Solver / Predictor Integration
// ----------------------------------------------------

// Vectorized trilinear interpolation for u, v, w using flat indexing for a specific viewport key
function getVelocityVectorAt(key, pz, py, px, outVelocity) {
    const fields = viewports[key].flowFields;
    if (!fields || !fields.u || !fields.v || !fields.w) {
        outVelocity[0] = 0.04;
        outVelocity[1] = 0.0;
        outVelocity[2] = 0.0;
        return;
    }
    
    px = Math.max(0, Math.min(NX - 1.001, px));
    py = Math.max(0, Math.min(NY - 1.001, py));
    pz = Math.max(0, Math.min(NZ - 1.001, pz));
    
    const x0 = Math.floor(px); const x1 = x0 + 1;
    const y0 = Math.floor(py); const y1 = y0 + 1;
    const z0 = Math.floor(pz); const z1 = z0 + 1;
    
    const xd = px - x0;
    const yd = py - y0;
    const zd = pz - z0;
    
    const sliceSize = NY * NX;
    const z0_offset = z0 * sliceSize;
    const z1_offset = z1 * sliceSize;
    const y0_offset = y0 * NX;
    const y1_offset = y1 * NX;
    
    const idx000 = z0_offset + y0_offset + x0;
    const idx100 = z0_offset + y0_offset + x1;
    const idx010 = z0_offset + y1_offset + x0;
    const idx110 = z0_offset + y1_offset + x1;
    const idx001 = z1_offset + y0_offset + x0;
    const idx101 = z1_offset + y0_offset + x1;
    const idx011 = z1_offset + y1_offset + x0;
    const idx111 = z1_offset + y1_offset + x1;
    
    const w000 = (1 - xd) * (1 - yd) * (1 - zd);
    const w100 = xd * (1 - yd) * (1 - zd);
    const w010 = (1 - xd) * yd * (1 - zd);
    const w110 = xd * yd * (1 - zd);
    const w001 = (1 - xd) * (1 - yd) * zd;
    const w101 = xd * (1 - yd) * zd;
    const w011 = (1 - xd) * yd * zd;
    const w111 = xd * yd * zd;
    
    // For U
    const u = fields.u;
    outVelocity[0] = u[idx000]*w000 + u[idx100]*w100 + u[idx010]*w010 + u[idx110]*w110 +
                     u[idx001]*w001 + u[idx101]*w101 + u[idx011]*w011 + u[idx111]*w111;
                     
    // For V
    const v = fields.v;
    outVelocity[1] = v[idx000]*w000 + v[idx100]*w100 + v[idx010]*w010 + v[idx110]*w110 +
                     v[idx001]*w001 + v[idx101]*w101 + v[idx011]*w011 + v[idx111]*w111;
                     
    // For W
    const w = fields.w;
    outVelocity[2] = w[idx000]*w000 + w[idx100]*w100 + w[idx010]*w010 + w[idx110]*w110 +
                     w[idx001]*w001 + w[idx101]*w101 + w[idx011]*w011 + w[idx111]*w111;
}

function initializeDefaultLaminarFlow() {
    const size = NZ * NY * NX;
    const makeDefaultFields = () => {
        const u = new Float32Array(size).fill(0.04);
        const v = new Float32Array(size);
        const w = new Float32Array(size);
        const p = new Float32Array(size);
        for (let z = 0; z < NZ; z++) {
            for (let y = 0; y < NY; y++) {
                for (let x = 0; x < NX; x++) {
                    if (obstacleMask[z][y][x] === 1) {
                        u[z * NY * NX + y * NX + x] = 0.0;
                    }
                }
            }
        }
        return { u, v, w, p, drag: null };
    };
    viewports.cfd.flowFields = makeDefaultFields();
    viewports.surrogate.flowFields = makeDefaultFields();
}

async function runSurrogatePredict() {
    setLoading("surrogate", true, "Neural Surrogate predicting flow fields...");
    
    // Convert current 3D obstacle mask to standard nested lists
    const maskList = [];
    for (let z = 0; z < NZ; z++) {
        const yList = [];
        for (let y = 0; y < NY; y++) {
            yList.push(Array.from(obstacleMask[z][y]));
        }
        maskList.push(yList);
    }
    
    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mask: maskList,
                Re: reynoldsNumber
            })
        });
        
        if (!response.ok) throw new Error("Predict request failed");
        
        // Parse the binary response as an ArrayBuffer
        const arrayBuffer = await response.arrayBuffer();
        const floatArray = new Float32Array(arrayBuffer);
        
        const fieldSize = NZ * NY * NX;
        const fields = viewports.surrogate.flowFields;
        fields.u = floatArray.subarray(0, fieldSize);
        fields.v = floatArray.subarray(fieldSize, fieldSize * 2);
        fields.w = floatArray.subarray(fieldSize * 2, fieldSize * 3);
        fields.p = floatArray.subarray(fieldSize * 3, fieldSize * 4);
        
        // Extract metrics from headers
        const drag = parseFloat(response.headers.get("X-Drag")) || 0.0;
        const execTime = parseInt(response.headers.get("X-Execution-Time-Ms")) || 0;
        fields.drag = drag;
        
        // Update display metrics
        document.getElementById("surrogateDragVal").innerText = drag.toFixed(4);
        document.getElementById("surrogateTimer").innerText = `${execTime} ms`;
        
        showToast("Instant 3D Neural inference finished!", "success");
        updateSliceTexture();
    } catch (err) {
        console.error(err);
        showToast("Error running surrogate prediction.", "error");
    } finally {
        setLoading("surrogate", false);
    }
}

async function runCFDNumerical() {
    setLoading("cfd", true, "Running numerical D3Q19 LBM simulation...");
    
    const maskList = [];
    for (let z = 0; z < NZ; z++) {
        const yList = [];
        for (let y = 0; y < NY; y++) {
            yList.push(Array.from(obstacleMask[z][y]));
        }
        maskList.push(yList);
    }
    
    try {
        const response = await fetch("/api/solve", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mask: maskList,
                Re: reynoldsNumber,
                steps: 150
            })
        });
        
        if (!response.ok) throw new Error("CFD Solve request failed");
        
        // Parse the binary response as an ArrayBuffer
        const arrayBuffer = await response.arrayBuffer();
        const floatArray = new Float32Array(arrayBuffer);
        
        const fieldSize = NZ * NY * NX;
        const fields = viewports.cfd.flowFields;
        fields.u = floatArray.subarray(0, fieldSize);
        fields.v = floatArray.subarray(fieldSize, fieldSize * 2);
        fields.w = floatArray.subarray(fieldSize * 2, fieldSize * 3);
        fields.p = floatArray.subarray(fieldSize * 3, fieldSize * 4);
        
        // Extract metrics from headers
        const drag = parseFloat(response.headers.get("X-Drag")) || 0.0;
        const execTime = parseInt(response.headers.get("X-Execution-Time-Ms")) || 0;
        fields.drag = drag;
        
        document.getElementById("cfdDragVal").innerText = drag.toFixed(4);
        document.getElementById("cfdTimer").innerText = `${execTime} ms`;
        
        showToast("CFD LBM solver convergence finished!", "success");
        updateSliceTexture();
    } catch (err) {
        console.error(err);
        showToast("Error executing CFD solver.", "error");
    } finally {
        setLoading("cfd", false);
    }
}



// ----------------------------------------------------
// Geometry & Mesh Building
// ----------------------------------------------------

// Reusable matrix for instanced mesh transforms
const _instanceMatrix = new THREE.Matrix4();

function updateVoxelViews() {
    const keys = ["cfd", "surrogate"];
    keys.forEach((key) => {
        const vp = viewports[key];
        
        // Dispose old instanced meshes
        while (vp.voxelGroup.children.length > 0) {
            const obj = vp.voxelGroup.children[0];
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
            vp.voxelGroup.remove(obj);
        }
        
        // Count solid voxels first so we can allocate InstancedMesh exactly
        let count = 0;
        for (let z = 0; z < NZ; z++)
            for (let y = 0; y < NY; y++)
                for (let x = 0; x < NX; x++)
                    if (obstacleMask[z][y][x] === 1) count++;
        
        if (count === 0) return;
        
        const opacity = renderMode === "particles" ? 0.45 : (renderMode === "slice" ? 0.25 : 1.0);
        
        const geom = new THREE.BoxGeometry(0.95, 0.95, 0.95);
        const mat = new THREE.MeshPhongMaterial({
            color: 0xfe0879,
            emissive: 0x330015,
            shininess: 90,
            transparent: opacity < 1.0,
            opacity: opacity
        });
        
        // Single InstancedMesh = 1 GPU draw call for all voxels
        const instancedMesh = new THREE.InstancedMesh(geom, mat, count);
        instancedMesh.castShadow = false;
        
        let idx = 0;
        for (let z = 0; z < NZ; z++) {
            for (let y = 0; y < NY; y++) {
                for (let x = 0; x < NX; x++) {
                    if (obstacleMask[z][y][x] === 1) {
                        _instanceMatrix.setPosition(x - NX / 2, y - NY / 2, z - NZ / 2);
                        instancedMesh.setMatrixAt(idx++, _instanceMatrix);
                    }
                }
            }
        }
        instancedMesh.instanceMatrix.needsUpdate = true;
        vp.voxelGroup.add(instancedMesh);
    });
    
    // Rebuild distance field for particle interaction coloring (O(1) per-frame lookup)
    buildObstacleDistanceField();
}

function updateSliceTexture() {
    if (renderMode !== "slice") return;
    
    const keys = ["cfd", "surrogate"];
    keys.forEach((key) => {
        const vp = viewports[key];
        if (!vp.flowFields || !vp.flowFields.u) return;
        vp.slicePlane.visible = true;
        
        let width, height;
        let dataArray;
        
        // Slice dimensions and position depends on plane axis choice
        if (sliceAxis === "z") {
            width = NX; height = NY;
            vp.slicePlane.geometry = new THREE.PlaneGeometry(NX, NY);
            vp.slicePlane.position.set(0, 0, sliceIndex - NZ / 2);
            vp.slicePlane.rotation.set(0, 0, 0);
            
            dataArray = new Uint8Array(NX * NY * 4);
            const z = Math.min(NZ - 1, Math.max(0, sliceIndex));
            const zOffset = z * NY * NX;
            for (let y = 0; y < NY; y++) {
                const yOffset = y * NX;
                for (let x = 0; x < NX; x++) {
                    const idx = (y * NX + x) * 4;
                    const idx_flow = zOffset + yOffset + x;
                    // Compute velocity magnitude
                    const u = vp.flowFields.u[idx_flow];
                    const v = vp.flowFields.v[idx_flow];
                    const w = vp.flowFields.w[idx_flow];
                    const mag = Math.sqrt(u*u + v*v + w*w);
                    
                    const col = getFlowColor(mag);
                    dataArray[idx] = col.r * 255;
                    dataArray[idx + 1] = col.g * 255;
                    dataArray[idx + 2] = col.b * 255;
                    dataArray[idx + 3] = obstacleMask[z][y][x] === 1 ? 50 : 255; // solid masking opacity
                }
            }
        } else if (sliceAxis === "y") {
            width = NX; height = NZ;
            vp.slicePlane.geometry = new THREE.PlaneGeometry(NX, NZ);
            vp.slicePlane.position.set(0, sliceIndex - NY / 2, 0);
            vp.slicePlane.rotation.set(Math.PI / 2, 0, 0);
            
            dataArray = new Uint8Array(NX * NZ * 4);
            const y = Math.min(NY - 1, Math.max(0, sliceIndex));
            const yOffset = y * NX;
            for (let z = 0; z < NZ; z++) {
                const zOffset = z * NY * NX;
                for (let x = 0; x < NX; x++) {
                    const idx = (z * NX + x) * 4;
                    const idx_flow = zOffset + yOffset + x;
                    const u = vp.flowFields.u[idx_flow];
                    const v = vp.flowFields.v[idx_flow];
                    const w = vp.flowFields.w[idx_flow];
                    const mag = Math.sqrt(u*u + v*v + w*w);
                    
                    const col = getFlowColor(mag);
                    dataArray[idx] = col.r * 255;
                    dataArray[idx + 1] = col.g * 255;
                    dataArray[idx + 2] = col.b * 255;
                    dataArray[idx + 3] = obstacleMask[z][y][x] === 1 ? 50 : 255;
                }
            }
        } else {
            width = NY; height = NZ;
            vp.slicePlane.geometry = new THREE.PlaneGeometry(NY, NZ);
            vp.slicePlane.position.set(sliceIndex - NX / 2, 0, 0);
            vp.slicePlane.rotation.set(0, Math.PI / 2, 0);
            
            dataArray = new Uint8Array(NY * NZ * 4);
            const x = Math.min(NX - 1, Math.max(0, sliceIndex));
            for (let z = 0; z < NZ; z++) {
                const zOffset = z * NY * NX;
                for (let y = 0; y < NY; y++) {
                    const idx = (z * NY + y) * 4;
                    const idx_flow = zOffset + y * NX + x;
                    const u = vp.flowFields.u[idx_flow];
                    const v = vp.flowFields.v[idx_flow];
                    const w = vp.flowFields.w[idx_flow];
                    const mag = Math.sqrt(u*u + v*v + w*w);
                    
                    const col = getFlowColor(mag);
                    dataArray[idx] = col.r * 255;
                    dataArray[idx + 1] = col.g * 255;
                    dataArray[idx + 2] = col.b * 255;
                    dataArray[idx + 3] = obstacleMask[z][y][x] === 1 ? 50 : 255;
                }
            }
        }
        
        // Generate texture and map to slice plane
        const texture = new THREE.DataTexture(dataArray, width, height, THREE.RGBAFormat);
        texture.needsUpdate = true;
        
        vp.slicePlane.material.map = texture;
        vp.slicePlane.material.needsUpdate = true;
    });
}

function hideSlicePlanes() {
    const keys = ["cfd", "surrogate"];
    keys.forEach((key) => {
        viewports[key].slicePlane.visible = false;
    });
}

// ----------------------------------------------------
// Presets Setup
// ----------------------------------------------------
function initPresets() {
    document.querySelectorAll(".btn-preset").forEach((btn) => {
        btn.addEventListener("click", (e) => {
            const presetName = e.currentTarget.getAttribute("data-preset");
            loadPreset(presetName);
            
            document.querySelectorAll(".btn-preset").forEach((b) => b.classList.remove("active"));
            e.currentTarget.classList.add("active");
            
            // Auto-predict flow around new preset shape
            runSurrogatePredict();
        });
    });
    
    document.getElementById("btnClearCanvas").addEventListener("click", () => {
        // Clear obstacle array
        for (let z = 0; z < NZ; z++) {
            for (let y = 0; y < NY; y++) {
                obstacleMask[z][y].fill(0);
            }
        }
        ctx2D.fillStyle = "#000000";
        ctx2D.fillRect(0, 0, NX, NY);
        
        updateVoxelViews();
        initializeDefaultLaminarFlow();
        document.getElementById("cfdDragVal").innerText = "--";
        document.getElementById("surrogateDragVal").innerText = "--";
        showToast("Design grid cleared.", "info");
    });
}

function loadPreset(name) {
    // Clear mask
    for (let z = 0; z < NZ; z++) {
        for (let y = 0; y < NY; y++) {
            obstacleMask[z][y].fill(0);
        }
    }
    
    const cz = 16, cy = 16, cx = 30;
    
    if (name === "sphere") {
        const radius = 4.5;
        for (let z = 0; z < NZ; z++) {
            for (let y = 0; y < NY; y++) {
                for (let x = 0; x < NX; x++) {
                    const distSq = (x-cx)**2 + (y-cy)**2 + (z-cz)**2;
                    if (distSq <= radius**2) {
                        obstacleMask[z][y][x] = 1;
                    }
                }
            }
        }
    } else if (name === "cylinder") {
        const radius = 3.5;
        // Extrude Y-axis centered cylinder (infinite along Z)
        for (let z = 0; z < NZ; z++) {
            for (let y = 0; y < NY; y++) {
                for (let x = 0; x < NX; x++) {
                    const distSq = (x-cx)**2 + (z-cz)**2;
                    if (distSq <= radius**2 && 4 <= y && y <= NY-5) {
                        obstacleMask[z][y][x] = 1;
                    }
                }
            }
        }
    } else if (name === "naca") {
        // Wing: Extruded NACA Profile along Z
        const chord = 22.0;
        const thickness = 0.12;
        // Symmetric airfoil shape
        for (let x_loc = 0; x_loc < chord; x_loc++) {
            const x_norm = x_loc / chord;
            const yt = 5 * thickness * (0.2969 * Math.sqrt(x_norm) - 0.126 * x_norm - 0.3516 * x_norm**2 + 0.2843 * x_norm**3 - 0.1015 * x_norm**4) * chord;
            const y_upper = cy + yt;
            const y_lower = cy - yt;
            
            const px = cx + x_loc;
            for (let y = Math.floor(y_lower); y <= Math.ceil(y_upper); y++) {
                for (let z = 5; z < NZ-5; z++) {
                    if (y >= 0 && y < NY) {
                        obstacleMask[z][y][px] = 1;
                    }
                }
            }
        }
    } else if (name === "heatsink") {
        // Double flat fin elements
        for (let z = 5; z < NZ-5; z++) {
            for (let y = 6; y < NY-6; y++) {
                // Fin 1
                obstacleMask[z][y][cx - 4] = 1;
                obstacleMask[z][y][cx - 3] = 1;
                // Fin 2
                obstacleMask[z][y][cx + 4] = 1;
                obstacleMask[z][y][cx + 5] = 1;
            }
        }
    }
    
    // Clear flow boundary zones for solver consistency
    for (let z = 0; z < NZ; z++) {
        for (let y = 0; y < NY; y++) {
            obstacleMask[z][y].fill(0, 0, 5); // Clear inlet
            obstacleMask[z][y].fill(0, NX - 5, NX); // Clear outlet
            obstacleMask[z][y][0] = 0;
            obstacleMask[z][y][NY - 1] = 0;
        }
        obstacleMask[0] = Array.from({ length: NY }, () => new Uint8Array(NX));
        obstacleMask[NZ - 1] = Array.from({ length: NY }, () => new Uint8Array(NX));
    }
    
    // Update local profile design canvas to match preset slice
    redraw2DCanvas();
    updateVoxelViews();
    initializeDefaultLaminarFlow();
}

function redraw2DCanvas() {
    ctx2D.fillStyle = "#000000";
    ctx2D.fillRect(0, 0, NX, NY);
    
    ctx2D.fillStyle = "#fe0879";
    // Render the center Z cross section of the mask onto the 2D workspace canvas
    const zCenter = 16;
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            if (obstacleMask[zCenter][y][x] === 1) {
                ctx2D.fillRect(x, y, 1, 1);
            }
        }
    }
}

// ----------------------------------------------------
// 2D Drawing Interface
// ----------------------------------------------------
function init2DDrawCanvas() {
    canvas2D.addEventListener("mousedown", (e) => {
        isDrawing = true;
        drawVoxel2D(e);
    });
    
    canvas2D.addEventListener("mousemove", (e) => {
        if (isDrawing) drawVoxel2D(e);
    });
    
    window.addEventListener("mouseup", () => {
        if (isDrawing) {
            isDrawing = false;
            updateVoxelViews();
            initializeDefaultLaminarFlow();
            // Auto-predict flow around newly drawn shape
            runSurrogatePredict();
        }
    });
    
    document.getElementById("toolBrush").addEventListener("click", (e) => {
        brushMode = "draw";
        document.querySelectorAll(".btn-tool").forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
    });
    
    document.getElementById("toolEraser").addEventListener("click", (e) => {
        brushMode = "erase";
        document.querySelectorAll(".btn-tool").forEach((b) => b.classList.remove("active"));
        e.currentTarget.classList.add("active");
    });
    
    document.getElementById("brushSize").addEventListener("input", (e) => {
        brushSize = parseInt(e.target.value);
    });
}

function drawVoxel2D(e) {
    const rect = canvas2D.getBoundingClientRect();
    const scaleX = NX / rect.width;
    const scaleY = NY / rect.height;
    
    const xMouse = Math.floor((e.clientX - rect.left) * scaleX);
    const yMouse = Math.floor((e.clientY - rect.top) * scaleY);
    
    // Prevent drawing on inflow/outflow boundary walls
    if (xMouse < 5 || xMouse >= NX - 5 || yMouse < 1 || yMouse >= NY - 1) return;
    
    const drawRadius = brushSize - 1;
    const zCenter = 16;
    
    for (let dy = -drawRadius; dy <= drawRadius; dy++) {
        for (let dx = -drawRadius; dx <= drawRadius; dx++) {
            const xCoord = xMouse + dx;
            const yCoord = yMouse + dy;
            
            // Limit checks
            if (xCoord >= 5 && xCoord < NX - 5 && yCoord >= 1 && yCoord < NY - 1) {
                const state = brushMode === "draw" ? 1 : 0;
                
                // Set obstacle voxel profile across the active center slices of Z axis
                // Replicate drawing from Z=6 to Z=25 (leaving margins free)
                for (let z = 6; z <= 25; z++) {
                    obstacleMask[z][yCoord][xCoord] = state;
                }
                
                // Visual update on the 2D workspace canvas
                ctx2D.fillStyle = state === 1 ? "#fe0879" : "#000000";
                ctx2D.fillRect(xCoord, yCoord, 1, 1);
            }
        }
    }
}

// ----------------------------------------------------
// App General Event Handlers
// ----------------------------------------------------
function initControls() {
    document.getElementById("reynoldsNum").addEventListener("input", (e) => {
        reynoldsNumber = parseInt(e.target.value);
        document.getElementById("reynoldsVal").innerText = reynoldsNumber;
    });
    
    document.getElementById("btnRunSurrogate").addEventListener("click", () => {
        runSurrogatePredict();
    });
    
    document.getElementById("btnRunCFD").addEventListener("click", () => {
        runCFDNumerical();
    });
    

    
    const renderSelect = document.getElementById("renderModeSelect");
    const sliceControls = document.getElementById("slicePlaneControls");
    
    renderSelect.addEventListener("change", (e) => {
        renderMode = e.target.value;
        
        if (renderMode === "slice") {
            sliceControls.style.display = "flex";
            updateSliceTexture();
        } else {
            sliceControls.style.display = "none";
            hideSlicePlanes();
        }
        
        updateVoxelViews();
    });
    
    const sliceAxisSelect = document.getElementById("sliceAxisSelect");
    const sliceSlider = document.getElementById("slicePlaneRange");
    
    sliceAxisSelect.addEventListener("change", (e) => {
        sliceAxis = e.target.value;
        // Adjust slider limits depending on plane selection axis
        if (sliceAxis === "x") {
            sliceSlider.max = NX - 1;
            sliceSlider.value = Math.floor(NX / 2);
        } else if (sliceAxis === "y") {
            sliceSlider.max = NY - 1;
            sliceSlider.value = Math.floor(NY / 2);
        } else {
            sliceSlider.max = NZ - 1;
            sliceSlider.value = Math.floor(NZ / 2);
        }
        sliceIndex = parseInt(sliceSlider.value);
        document.getElementById("slicePlaneValue").innerText = sliceIndex;
        updateSliceTexture();
    });
    
    sliceSlider.addEventListener("input", (e) => {
        sliceIndex = parseInt(e.target.value);
        document.getElementById("slicePlaneValue").innerText = sliceIndex;
        updateSliceTexture();
    });
}

// ----------------------------------------------------
// UI Auxiliaries (Toast, Status, Loader)
// ----------------------------------------------------
async function checkModelStatus() {
    try {
        const response = await fetch("/api/model_status");
        const data = await response.json();
        
        const dot = document.getElementById("surrogateStatusDot");
        const txt = document.getElementById("surrogateStatusText");
        
        if (data.surrogate_loaded) {
            dot.className = "status-dot success";
            txt.innerText = "Loaded (Ready)";
        } else {
            dot.className = "status-dot warning";
            txt.innerText = "Weights Missing";
            showToast("Surrogate weights missing locally! Run Colab export.", "warning");
        }
    } catch (err) {
        console.error(err);
        const dot = document.getElementById("surrogateStatusDot");
        const txt = document.getElementById("surrogateStatusText");
        dot.className = "status-dot warning";
        txt.innerText = "Offline";
    }
}

function showToast(msg, type = "info") {
    const toast = document.getElementById("toast");
    toast.innerText = msg;
    toast.className = `toast show ${type}`;
    
    // Auto hide after 3 seconds
    setTimeout(() => {
        toast.classList.remove("show");
    }, 3000);
}

function setLoading(key, state, text = "") {
    const overlay = document.getElementById(`${key}Loading`);
    const label = document.getElementById(`${key}LoadingText`);
    
    if (state) {
        overlay.style.opacity = "1";
        overlay.style.pointerEvents = "all";
        if (label) label.innerText = text;
    } else {
        overlay.style.opacity = "0";
        overlay.style.pointerEvents = "none";
    }
}





// ----------------------------------------------------
// Animation Frame Loop
// ----------------------------------------------------
// Preallocated temporary array to prevent GC overhead in animation loop
const velVec = new Float32Array(3);

function animate() {
    requestAnimationFrame(animate);
    
    const keys = ["cfd", "surrogate"];
    keys.forEach((key) => {
        const vp = viewports[key];
        
        // 1. Render Streamline Particles flow
        if (renderMode === "particles" && vp.flowFields && vp.flowFields.u) {
            vp.particles.visible = true;
            
            const positions = vp.particles.geometry.attributes.position.array;
            const colors = vp.particles.geometry.attributes.color.array;
            const size = vp.particleData.length;
            
            for (let i = 0; i < size; i++) {
                const p = vp.particleData[i];
                
                // Sample 3D local velocities using vectorized trilinear interpolation
                getVelocityVectorAt(key, p.z, p.y, p.x, velVec);
                const vx = velVec[0];
                const vy = velVec[1];
                const vz = velVec[2];
                
                // Update position using Euler advection step
                const dt = 6.0;
                p.x += vx * dt;
                p.y += vy * dt;
                p.z += vz * dt;
                
                // Shift history buffer
                p.history.shift();
                p.history.push({ x: p.x, y: p.y, z: p.z });
                
                // Recolor: blend speed color with hot interaction glow near obstacle
                const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
                const proximity = getObstacleProximity(p.x, p.y, p.z);
                const col = getInteractionColor(speed, proximity);
                
                const idx = i * 6;
                
                // Set fading segment colors (tip -> body -> head)
                const fadeFactors = [0.15, 0.15, 0.45, 0.45, 1.0, 1.0];
                for (let j = 0; j < 6; j++) {
                    const f = fadeFactors[j];
                    colors[(idx + j) * 3]     = col.r * f;
                    colors[(idx + j) * 3 + 1] = col.g * f;
                    colors[(idx + j) * 3 + 2] = col.b * f;
                }
                
                p.life -= 0.55;
                
                // Reset particles if they exit, get stuck inside obstacle, or expire
                const isOutOfBounds = p.x >= NX - 5 || p.x < 1 || p.y >= NY - 1 || p.y < 1 || p.z >= NZ - 1 || p.z < 1;
                let isInsideObstacle = false;
                if (!isOutOfBounds) {
                    const zIdx = Math.floor(p.z);
                    const yIdx = Math.floor(p.y);
                    const xIdx = Math.floor(p.x);
                    const speed2 = vx*vx + vy*vy + vz*vz;
                    isInsideObstacle = (speed2 < 0.0001) && (obstacleMask[zIdx][yIdx][xIdx] === 1);
                }
                
                if (isOutOfBounds || p.life <= 0 || isInsideObstacle) {
                    p.x = 1.0 + Math.random() * 4.0;
                    p.y = 1.0 + Math.random() * (NY - 2.0);
                    p.z = 1.0 + Math.random() * (NZ - 2.0);
                    p.life = 450 + Math.random() * 200;
                    
                    // Reset history buffer to prevent cross-viewport stretching lines
                    for (let h = 0; h < 4; h++) {
                        p.history[h] = { x: p.x, y: p.y, z: p.z };
                    }
                }
                
                // Update vertex positions for Segment 3 (tip)
                positions[idx * 3]         = p.history[0].x - NX / 2;
                positions[idx * 3 + 1]     = p.history[0].y - NY / 2;
                positions[idx * 3 + 2]     = p.history[0].z - NZ / 2;
                positions[(idx + 1) * 3]     = p.history[1].x - NX / 2;
                positions[(idx + 1) * 3 + 1] = p.history[1].y - NY / 2;
                positions[(idx + 1) * 3 + 2] = p.history[1].z - NZ / 2;
                
                // Update vertex positions for Segment 2 (body)
                positions[(idx + 2) * 3]     = p.history[1].x - NX / 2;
                positions[(idx + 2) * 3 + 1] = p.history[1].y - NY / 2;
                positions[(idx + 2) * 3 + 2] = p.history[1].z - NZ / 2;
                positions[(idx + 3) * 3]     = p.history[2].x - NX / 2;
                positions[(idx + 3) * 3 + 1] = p.history[2].y - NY / 2;
                positions[(idx + 3) * 3 + 2] = p.history[2].z - NZ / 2;
                
                // Update vertex positions for Segment 1 (head)
                positions[(idx + 4) * 3]     = p.history[2].x - NX / 2;
                positions[(idx + 4) * 3 + 1] = p.history[2].y - NY / 2;
                positions[(idx + 4) * 3 + 2] = p.history[2].z - NZ / 2;
                positions[(idx + 5) * 3]     = p.history[3].x - NX / 2;
                positions[(idx + 5) * 3 + 1] = p.history[3].y - NY / 2;
                positions[(idx + 5) * 3 + 2] = p.history[3].z - NZ / 2;
            }
            
            vp.particles.geometry.attributes.position.needsUpdate = true;
            vp.particles.geometry.attributes.color.needsUpdate = true;
        } else {
            vp.particles.visible = false;
        }
        
        // 2. Refresh viewport controls & render
        vp.controls.update();
        vp.renderer.render(vp.scene, vp.camera);
    });
}
