process.env.ELECTRON_DISABLE_SECURITY_WARNINGS = 'true';
const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const http = require('http')
const fs = require('fs')

let mainWindow
let backendProcess = null
let staticServer = null

const isDev = !app.isPackaged && process.env.NODE_ENV !== 'production';

function startBackend() {
  if (isDev) return;
  // backend_api is copied as extraResource
  const backendExe = path.join(process.resourcesPath, 'backend_api', 'backend_api.exe');
  
  backendProcess = spawn(backendExe, [], {
    cwd: path.dirname(backendExe),
    stdio: 'ignore',
    shell: false
  });
  
  backendProcess.on('error', (err) => {
    console.error("Failed to start backend process:", err);
  });
}

function startStaticServer() {
  if (isDev) return;

  const mimeTypes = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'text/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg'
  };

  staticServer = http.createServer((req, res) => {
    const urlPath = req.url.split('?')[0];
    const distPath = path.join(__dirname, 'dist');
    let filePath = path.join(distPath, urlPath === '/' ? 'index.html' : urlPath);
    
    if (!filePath.startsWith(distPath)) {
      res.statusCode = 403;
      res.end('Forbidden');
      return;
    }

    fs.readFile(filePath, (err, data) => {
      if (err) {
        fs.readFile(path.join(distPath, 'index.html'), (err2, data2) => {
          if (err2) {
            res.statusCode = 404;
            res.end('Not Found');
          } else {
            res.writeHead(200, { 'Content-Type': 'text/html' });
            res.end(data2);
          }
        });
      } else {
        const ext = path.extname(filePath).toLowerCase();
        const contentType = mimeTypes[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': contentType });
        res.end(data);
      }
    });
  });

  staticServer.listen(5173, '0.0.0.0', () => {
    console.log("Exposing app via localhost:5173");
  });
}

let splashWindow = null

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 500,
    height: 400,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    resizable: false,
    center: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    }
  })
  splashWindow.loadFile(path.join(__dirname, 'splash.html'))
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: isDev, // show immediately in dev, hidden for splash in production
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      nodeIntegration: false,
      contextIsolation: true
    },
    autoHideMenuBar: true
  })

  if (!isDev) {
    mainWindow.once('ready-to-show', () => {
      // Keep splash for a short moment (1.5s) to let server bind and feel premium
      setTimeout(() => {
        if (splashWindow) {
          splashWindow.close()
        }
        mainWindow.show()
      }, 1500)
    })
  }

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    // Wait slightly to ensure static server is ready
    setTimeout(() => {
      mainWindow.loadURL('http://localhost:5173')
    }, 500)
  }
}

app.whenReady().then(() => {
  if (!isDev) {
    createSplashWindow()
  }
  startBackend()
  startStaticServer()
  createWindow()

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit()
})

app.on('will-quit', () => {
  if (backendProcess) {
    try {
      backendProcess.kill();
    } catch (e) {
      console.error("Error killing backend process:", e);
    }
  }
  if (staticServer) {
    staticServer.close();
  }
})

// Handle IPC
ipcMain.handle('dialog:openFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Models', extensions: ['onnx', 'pth'] }
    ]
  })
  if (canceled) {
    return null
  } else {
    return filePaths[0]
  }
})

