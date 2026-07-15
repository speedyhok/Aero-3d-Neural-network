// ==========================================
// Developer: Mohibul Hoque
// Email: hokworks@gmail.com
// LinkedIn: linkedin.com/in/speedymohibul
// ==========================================

// CFD Flow Visualization Frontend Application Logic

// Grid coordinates
const NX = 120;
const NY = 40;

// Application State
let obstacleMask = Array(NY).fill().map(() => Array(NX).fill(0));
let activePreset = 'cylinder';
let brushSize = 5;
let isDrawing = false;
let vizMode = 'velocity'; // 'velocity' or 'pressure'

// Velocity/Pressure fields for Solver
let cfdFields = null;

// Particle System for Streamline Animation
let particles = [];
const NUM_PARTICLES = 180;
let animationId = null;

// DOM Elements
const designCanvas = document.getElementById('designCanvas');
const designCtx = designCanvas.getContext('2d');

const cfdCanvas = document.getElementById('cfdCanvas');
const cfdCtx = cfdCanvas.getContext('2d');
const cfdLoading = document.getElementById('cfdLoading');
const cfdTimer = document.getElementById('cfdTimer');

// Offscreen cache canvas for static background heatmap
const heatmapCacheCanvas = document.createElement('canvas');
heatmapCacheCanvas.width = NX;
heatmapCacheCanvas.height = NY;
const heatmapCacheCtx = heatmapCacheCanvas.getContext('2d');
let cacheNeedsUpdate = true;

// UI Controls
const reynoldsNum = document.getElementById('reynoldsNum');
const reynoldsVal = document.getElementById('reynoldsVal');
const brushSizeSlider = document.getElementById('brushSize');
const brushSizeVal = document.getElementById('brushSizeVal');
const btnRunSolver = document.getElementById('btnRunSolver');
const btnClearCanvas = document.getElementById('btnClearCanvas');

// Toast Notification
const toast = document.getElementById('toast');

// ----------------------------------------------------
// Toast Alert Helper
// ----------------------------------------------------
function showToast(message, isError = false) {
    toast.textContent = message;
    toast.style.borderColor = isError ? 'var(--accent-red)' : 'var(--accent-cyan)';
    toast.style.boxShadow = isError 
        ? '0 10px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(239, 68, 68, 0.25)' 
        : '0 10px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(6, 182, 212, 0.25)';
    toast.classList.add('active');
    setTimeout(() => {
        toast.classList.remove('active');
    }, 4000);
}

// ----------------------------------------------------
// Presets drawing
// ----------------------------------------------------
function applyGeometryPreset(type) {
    // Reset mask
    obstacleMask = Array(NY).fill().map(() => Array(NX).fill(0));
    
    const cy = Math.round(NY / 2);
    const cx = 30; // Obstacle center X
    
    if (type === 'cylinder') {
        const radius = 4;
        for (let y = 0; y < NY; y++) {
            for (let x = 0; x < NX; x++) {
                if ((x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2) {
                    obstacleMask[y][x] = 1;
                }
            }
        }
    } else if (type === 'heatsink') {
        // Draw standard vertical cooling fin (rectangle)
        const width = 8;
        const height = 12;
        for (let y = 0; y < NY; y++) {
            for (let x = 0; x < NX; x++) {
                if (Math.abs(x - cx) <= width/2 && Math.abs(y - cy) <= height/2) {
                    obstacleMask[y][x] = 1;
                }
            }
        }
    } else if (type === 'airfoil') {
        // Generate NACA 2412 coordinates on workspace grid
        const chord = 22;
        const AoA = 8; // Angle of attack in degrees
        const camber = 0.04;
        const camberPos = 0.4;
        const thickness = 0.12;
        
        generateNacaAirfoil(chord, AoA, camber, camberPos, thickness, cx, cy);
    }
    
    drawDesignWorkspace();
}

function generateNacaAirfoil(chord, AoA_deg, camber, camberPos, thickness, cx, cy) {
    const AoA = AoA_deg * Math.PI / 180.0;
    const nPoints = 100;
    
    let xChord = [];
    for (let i = 0; i <= nPoints; i++) xChord.push(i / nPoints);
    
    let yc = Array(nPoints + 1).fill(0);
    let dyc_dx = Array(nPoints + 1).fill(0);
    
    const m = camber;
    const p = camberPos;
    
    for (let i = 0; i <= nPoints; i++) {
        let x = xChord[i];
        if (x < p) {
            if (p > 0) {
                yc[i] = (m / (p**2)) * (2 * p * x - x**2);
                dyc_dx[i] = (2 * m / (p**2)) * (p - x);
            }
        } else {
            if (p < 1) {
                yc[i] = (m / ((1 - p)**2)) * ((1 - 2 * p) + 2 * p * x - x**2);
                dyc_dx[i] = (2 * m / ((1 - p)**2)) * (p - x);
            }
        }
    }
    
    // Upper and lower surfaces coordinates relative to leading edge [0,0]
    let xu = Array(nPoints + 1);
    let yu = Array(nPoints + 1);
    let xl = Array(nPoints + 1);
    let yl = Array(nPoints + 1);
    
    const t = thickness;
    
    for (let i = 0; i <= nPoints; i++) {
        let x = xChord[i];
        let theta = Math.atan(dyc_dx[i]);
        let yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4);
        
        xu[i] = x - yt * Math.sin(theta);
        yu[i] = yc[i] + yt * Math.cos(theta);
        xl[i] = x + Math.sin(theta) * yt;
        yl[i] = yc[i] - Math.cos(theta) * yt;
    }
    
    // Scale and rotate coordinates
    let polyPoints = [];
    const cosA = Math.cos(AoA);
    const sinA = Math.sin(AoA);
    
    function rotateAndTranslate(x, y) {
        let xs = x * chord;
        let ys = y * chord;
        let xr = xs * cosA - ys * sinA + cx;
        let yr = xs * sinA + ys * cosA + cy;
        return [xr, yr];
    }
    
    for (let i = 0; i <= nPoints; i++) {
        polyPoints.push(rotateAndTranslate(xu[i], yu[i]));
    }
    for (let i = nPoints; i >= 0; i--) {
        polyPoints.push(rotateAndTranslate(xl[i], yl[i]));
    }
    
    // Fill the mask grid cells using ray casting algorithm inside polygon bounds
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            if (isPointInPolygon([x, y], polyPoints)) {
                obstacleMask[y][x] = 1;
            }
        }
    }
}

function isPointInPolygon(point, polygon) {
    const x = point[0], y = point[1];
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// ----------------------------------------------------
// Workspace drawing render
// ----------------------------------------------------
function drawDesignWorkspace() {
    designCtx.fillStyle = '#020409';
    designCtx.fillRect(0, 0, NX, NY);
    
    // Draw cells
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            if (obstacleMask[y][x] === 1) {
                // Glowy neon-magenta/red for obstacle cells
                designCtx.fillStyle = '#ef4444';
                designCtx.fillRect(x, y, 1, 1);
            }
        }
    }
}

// ----------------------------------------------------
// Heatmap Color-Mapping
// ----------------------------------------------------
function getColormapColor(val, min, max, mode) {
    let norm = (val - min) / (max - min || 1.0);
    norm = Math.max(0.0, Math.min(1.0, norm));
    
    if (mode === 'velocity') {
        // Cold-Hot colormap: Blue -> Cyan -> Green -> Yellow -> Red
        let r = 0, g = 0, b = 0;
        if (norm < 0.25) {
            r = 0;
            g = Math.round(norm * 4 * 255);
            b = 255;
        } else if (norm < 0.5) {
            r = 0;
            g = 255;
            b = Math.round((0.5 - norm) * 4 * 255);
        } else if (norm < 0.75) {
            r = Math.round((norm - 0.5) * 4 * 255);
            g = 255;
            b = 0;
        } else {
            r = 255;
            g = Math.round((1.0 - norm) * 4 * 255);
            b = 0;
        }
        return [r, g, b];
    } else {
        // Pressure mode: Low Pressure: Dark Blue/Slate, High Pressure: Electric Violet/Magenta
        let r = Math.round(10 + norm * 215);
        let g = Math.round(30 - norm * 20);
        let b = Math.round(80 + norm * 170);
        return [r, g, b];
    }
}

function renderFieldHeatmap(canvas, data, mask, mode) {
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(NX, NY);
    
    // Find min and max for normalizing values
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            if (mask[y][x] === 0) {
                let v = data[y][x];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
        }
    }
    
    // Handle edge case of flat fields
    if (minVal === maxVal) {
        minVal -= 0.1;
        maxVal += 0.1;
    }
    
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            const pixelIdx = (y * NX + x) * 4;
            
            if (mask[y][x] === 1) {
                // Obstacle is rendered as solid black
                imgData.data[pixelIdx] = 2;
                imgData.data[pixelIdx + 1] = 4;
                imgData.data[pixelIdx + 2] = 9;
                imgData.data[pixelIdx + 3] = 255;
            } else {
                const rgb = getColormapColor(data[y][x], minVal, maxVal, mode);
                imgData.data[pixelIdx] = rgb[0];
                imgData.data[pixelIdx + 1] = rgb[1];
                imgData.data[pixelIdx + 2] = rgb[2];
                imgData.data[pixelIdx + 3] = 255;
            }
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// ----------------------------------------------------
// Streamline Particle Simulator (Wind Tunnel smoke)
// ----------------------------------------------------
class StreamlineParticle {
    constructor() {
        this.reset();
        // Stagger initial x-positions to distribute them nicely across the grid
        this.x = Math.random() * NX;
    }
    
    reset() {
        this.x = 0;
        this.y = Math.random() * NY;
        this.life = 100 + Math.random() * 150;
        this.speedScale = 0.8;
    }
    
    update(uField, vField, mask) {
        this.life -= 1;
        if (this.life <= 0 || this.x >= NX - 1 || this.x < 0 || this.y >= NY - 1 || this.y < 0) {
            this.reset();
            return;
        }
        
        // Bilinear interpolation or simple nearest neighbor lookup
        const xi = Math.floor(this.x);
        const yi = Math.floor(this.y);
        
        if (mask[yi][xi] === 1) {
            this.reset();
            return;
        }
        
        const u = uField[yi][xi];
        const v = vField[yi][xi];
        
        this.x += u * this.speedScale;
        this.y += v * this.speedScale;
    }
}

function initializeParticles() {
    particles = [];
    for (let i = 0; i < NUM_PARTICLES; i++) {
        particles.push(new StreamlineParticle());
    }
}

function animateStreamlines() {
    if (!cfdFields) {
        animationId = requestAnimationFrame(animateStreamlines);
        return;
    }
    
    if (cacheNeedsUpdate) {
        const activeCfdData = vizMode === 'velocity' ? cfdFields.vel_mag : cfdFields.p;
        renderFieldHeatmap(heatmapCacheCanvas, activeCfdData, obstacleMask, vizMode);
        cacheNeedsUpdate = false;
    }
    
    // Clear and draw cached background
    cfdCtx.drawImage(heatmapCacheCanvas, 0, 0);
    
    drawParticlesOnCanvas(cfdCanvas, cfdFields.u, cfdFields.v);
    
    animationId = requestAnimationFrame(animateStreamlines);
}

function drawParticlesOnCanvas(canvas, uField, vField) {
    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 1;
    
    particles.forEach(p => {
        // Save previous position
        const prevX = p.x;
        const prevY = p.y;
        
        p.update(uField, vField, obstacleMask);
        
        // Draw line connecting previous to current position
        // Only draw if it wasn't just reset to 0
        if (p.x > 0.5 && prevX < p.x) {
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(p.x, p.y);
            ctx.stroke();
        }
    });
}

// ----------------------------------------------------
// API Communication
// ----------------------------------------------------
async function runCFDSolver() {
    cfdLoading.classList.add('active');
    cfdTimer.textContent = 'Calculating...';
    
    const t0 = performance.now();
    try {
        const response = await fetch('/api/solve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mask: obstacleMask,
                Re: parseFloat(reynoldsNum.value),
                steps: 2500
            })
        });
        
        if (!response.ok) throw new Error('Solver error occurred.');
        const data = await response.json();
        
        const t1 = performance.now();
        const duration = ((t1 - t0) / 1000).toFixed(2);
        
        // Calculate velocity magnitude field: sqrt(u^2 + v^2)
        const u = data.u;
        const v = data.v;
        const p = data.p;
        let vel_mag = Array(NY).fill().map(() => Array(NX).fill(0));
        
        for (let y = 0; y < NY; y++) {
            for (let x = 0; x < NX; x++) {
                vel_mag[y][x] = Math.sqrt(u[y][x]**2 + v[y][x]**2);
            }
        }
        
        cfdFields = { u, v, p, vel_mag };
        cacheNeedsUpdate = true;
        cfdTimer.textContent = `${duration} s`;
        showToast('Numerical CFD Simulation completed!');
        
    } catch (e) {
        showToast(e.message || 'Failed to connect to numerical solver.', true);
        cfdTimer.textContent = 'Error';
    } finally {
        cfdLoading.classList.remove('active');
    }
}

// ----------------------------------------------------
// UI Event Bindings & Drawing Logic
// ----------------------------------------------------
function getCanvasMouseCoords(e) {
    const rect = designCanvas.getBoundingClientRect();
    // Scale coords back to 120x40 internal representation
    const x = Math.floor((e.clientX - rect.left) / rect.width * NX);
    const y = Math.floor((e.clientY - rect.top) / rect.height * NY);
    return [
        Math.max(0, Math.min(NX - 1, x)),
        Math.max(0, Math.min(NY - 1, y))
    ];
}

function drawCircleBrush(cx, cy, radius, drawVal) {
    for (let y = 0; y < NY; y++) {
        for (let x = 0; x < NX; x++) {
            if ((x - cx)**2 + (y - cy)**2 <= radius**2) {
                // Keep some margins clear to prevent choking boundary conditions
                if (x > 5 && x < NX - 6 && y > 1 && y < NY - 2) {
                    obstacleMask[y][x] = drawVal;
                }
            }
        }
    }
}

// Mouse Event listeners on design board
designCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const [x, y] = getCanvasMouseCoords(e);
    // Draw: left click draws obstacle (1), right click / shift-click erases (0)
    const drawVal = (e.shiftKey || e.button === 2) ? 0 : 1;
    drawCircleBrush(x, y, brushSize / 2, drawVal);
    drawDesignWorkspace();
});

designCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const [x, y] = getCanvasMouseCoords(e);
    const drawVal = (e.shiftKey || e.button === 2) ? 0 : 1;
    drawCircleBrush(x, y, brushSize / 2, drawVal);
    drawDesignWorkspace();
});

window.addEventListener('mouseup', () => {
    isDrawing = false;
});

// Prevent right-click context menu on design canvas
designCanvas.addEventListener('contextmenu', e => e.preventDefault());

// Preset clicks
document.querySelectorAll('.btn-preset[data-preset]').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activePreset = btn.getAttribute('data-preset');
        applyGeometryPreset(activePreset);
    });
});

btnClearCanvas.addEventListener('click', () => {
    obstacleMask = Array(NY).fill().map(() => Array(NX).fill(0));
    drawDesignWorkspace();
    cfdFields = null;
    cacheNeedsUpdate = true;
    
    // Clear display canvas
    cfdCtx.clearRect(0, 0, NX, NY);
    cfdTimer.textContent = '--';
});

// Slider values updating
reynoldsNum.addEventListener('input', () => {
    reynoldsVal.textContent = reynoldsNum.value;
});
brushSizeSlider.addEventListener('input', () => {
    brushSize = parseInt(brushSizeSlider.value);
    brushSizeVal.textContent = `${brushSize} px`;
});

// Run buttons
btnRunSolver.addEventListener('click', runCFDSolver);

// Viewport mode triggers (Velocity vs Pressure)
document.querySelectorAll('.btn-viz').forEach(btn => {
    btn.addEventListener('click', () => {
        const mode = btn.getAttribute('data-mode');
        vizMode = mode;
        cacheNeedsUpdate = true;
        
        // Sync active state on both viewports for visual consistency
        document.querySelectorAll('.btn-viz').forEach(b => {
            if (b.getAttribute('data-mode') === mode) {
                b.classList.add('active');
            } else {
                b.classList.remove('active');
            }
        });
    });
});

// ----------------------------------------------------
// Initialisation
// ----------------------------------------------------
function init() {
    applyGeometryPreset('cylinder');
    initializeParticles();
    animateStreamlines();
}

window.onload = init;
