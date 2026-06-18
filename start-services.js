const { spawn, exec } = require('child_process');
const path = require('path');

console.log("===================================================");
console.log("  Auditory AI Integrated Runner");
console.log("===================================================\n");

// Keep track of processes
const processes = [];

function registerProcess(proc, name) {
  processes.push({ proc, name });
  
  proc.on('error', (err) => {
    console.error(`[${name}] Failed to start:`, err);
  });
}

// 1. Spawn FastAPI Python backend (Disabled - client-side WASM inference active)
// console.log("[System] Starting Python Backend...");
// const backendPath = path.join(__dirname, 'backend');
// const pythonExec = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
// const backend = spawn(pythonExec, ['main.py'], {
//   cwd: backendPath,
//   stdio: 'inherit',
//   shell: true
// });
// registerProcess(backend, "Backend");

// 2. Spawn Vite dev server
console.log("[System] Starting Vite Dev Server...");
const frontendPath = path.join(__dirname, 'frontend');
const vite = spawn('node', ['node_modules/vite/bin/vite.js'], {
  cwd: frontendPath,
  stdio: 'inherit',
  shell: true
});
registerProcess(vite, "Vite");

// 3. Spawn Electron App & Open Browser after 2.5 seconds (when Vite has bound)
console.log("[System] Launching App and Browser in 2.5 seconds...");
let electronProc = null;

setTimeout(() => {
  console.log("[System] Starting Electron...");
  electronProc = spawn('node', ['node_modules/electron/cli.js', '.'], {
    cwd: frontendPath,
    stdio: 'inherit',
    shell: true
  });
  registerProcess(electronProc, "Electron");
  
  console.log("[System] Opening browser tab...");
  exec('start http://localhost:5173');
  
  electronProc.on('close', (code) => {
    console.log(`[System] Electron exited. Cleaning up all services...`);
    cleanup();
  });
}, 2500);

// Helper to clean up all spawned processes on exit
let isCleaningUp = false;
function cleanup() {
  if (isCleaningUp) return;
  isCleaningUp = true;
  console.log("\n[System] Shutting down all running services...");
  
  processes.forEach(({ proc, name }) => {
    try {
      if (proc && !proc.killed) {
        console.log(`[System] Stopping ${name}...`);
        proc.kill('SIGTERM');
      }
    } catch (e) {
      // ignore kill errors
    }
  });
  
  // Failsafe exit
  setTimeout(() => {
    process.exit(0);
  }, 1000);
}

// Bind process events
process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
process.on('exit', cleanup);
