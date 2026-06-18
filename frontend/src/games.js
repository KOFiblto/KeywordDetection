// Retro Audio Synth for game sound effects
let audioSynthCtx = null;

// Helper for backward compatibility with older browsers/electron context.roundRect
function drawRoundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    if (typeof ctx.roundRect === 'function') {
        ctx.roundRect(x, y, w, h, r);
    } else {
        if (w < 2 * r) r = w / 2;
        if (h < 2 * r) r = h / 2;
        ctx.moveTo(x + r, y);
        ctx.arcTo(x + w, y, x + w, y + h, r);
        ctx.arcTo(x + w, y + h, x, y + h, r);
        ctx.arcTo(x, y + h, x, y, r);
        ctx.arcTo(x, y, x + w, y, r);
        ctx.closePath();
    }
}

function playSound(type) {
    try {
        if (!audioSynthCtx) {
            audioSynthCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        if (audioSynthCtx.state === 'suspended') {
            audioSynthCtx.resume();
        }
        const osc = audioSynthCtx.createOscillator();
        const gain = audioSynthCtx.createGain();
        osc.connect(gain);
        gain.connect(audioSynthCtx.destination);
        
        const now = audioSynthCtx.currentTime;
        if (type === 'flap') {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(150, now);
            osc.frequency.exponentialRampToValueAtTime(380, now + 0.1);
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
            osc.start(now);
            osc.stop(now + 0.1);
        } else if (type === 'score') {
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(523.25, now); // C5
            osc.frequency.setValueAtTime(659.25, now + 0.08); // E5
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.22);
            osc.start(now);
            osc.stop(now + 0.22);
        } else if (type === 'hit') {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(120, now);
            osc.frequency.exponentialRampToValueAtTime(30, now + 0.35);
            gain.gain.setValueAtTime(0.18, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
            osc.start(now);
            osc.stop(now + 0.35);
        } else if (type === 'laser') {
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(700, now);
            osc.frequency.exponentialRampToValueAtTime(150, now + 0.12);
            gain.gain.setValueAtTime(0.05, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.12);
            osc.start(now);
            osc.stop(now + 0.12);
        } else if (type === 'shield') {
            osc.type = 'sine';
            osc.frequency.setValueAtTime(250, now);
            osc.frequency.linearRampToValueAtTime(320, now + 0.15);
            osc.frequency.linearRampToValueAtTime(250, now + 0.3);
            gain.gain.setValueAtTime(0.1, now);
            gain.gain.linearRampToValueAtTime(0.1, now + 0.22);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.3);
            osc.start(now);
            osc.stop(now + 0.3);
        } else if (type === 'shield_blocked') {
            osc.type = 'square';
            osc.frequency.setValueAtTime(280, now);
            osc.frequency.setValueAtTime(220, now + 0.08);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
            osc.start(now);
            osc.stop(now + 0.15);
        }
    } catch (e) {
        console.error("Audio synth error:", e);
    }
}

// ==========================================
// 1. FLAPPY BIRD GAME CLASS
// ==========================================
class FlappyBird {
    constructor(canvas, onScore, onGameOver) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onScore = onScore;
        this.onGameOver = onGameOver;
        
        this.bird = { x: 80, y: 180, radius: 14, velocity: 0, gravity: 0.0003, jump: -0.16 };
        this.pipes = [];
        this.pipeWidth = 60;
        this.pipeGap = 185;
        this.pipeSpeed = 0.25;
        this.frameCount = 0;
        this.score = 0;
        this.isPlaying = false;
        this.particles = [];
    }
    
    start() {
        this.bird.y = 180;
        this.bird.velocity = 0;
        this.pipes = [];
        this.score = 0;
        this.isPlaying = true;
        this.frameCount = 0;
        this.pipeSpawnTimer = 0;
        this.particles = [];
        this.spawnPipe();
    }
    
    spawnPipe() {
        const minHeight = 50;
        const maxHeight = this.canvas.height - this.pipeGap - minHeight;
        const topHeight = Math.floor(Math.random() * (maxHeight - minHeight + 1)) + minHeight;
        this.pipes.push({
            x: this.canvas.width,
            topHeight: topHeight,
            bottomHeight: this.canvas.height - topHeight - this.pipeGap,
            passed: false
        });
    }
    
    handleCommand(cmd) {
        if (!this.isPlaying) return;
        if (cmd === 'UP') {
            this.bird.velocity = this.bird.jump;
            playSound('flap');
            for (let i = 0; i < 6; i++) {
                this.particles.push({
                    x: this.bird.x - 8,
                    y: this.bird.y,
                    vx: -Math.random() * 2 - 1,
                    vy: (Math.random() - 0.5) * 4,
                    alpha: 1.0,
                    size: Math.random() * 3 + 2,
                    color: 'rgba(255, 255, 255, 0.6)'
                });
            }
        }
    }
    
    update() {
        if (!this.isPlaying) return;
        
        const speed = window.gameSpeedMultiplier || 1.0;
        this.frameCount += speed;
        
        this.bird.velocity += this.bird.gravity * speed;
        // Cap falling velocity for slower, gentler descent
        if (this.bird.velocity > 0.04) {
            this.bird.velocity = 0.04;
        }
        this.bird.y += this.bird.velocity * speed;
        
        // Ground and ceiling crash check
        if (this.bird.y + this.bird.radius > this.canvas.height - 25) {
            this.gameOver();
            return;
        }
        if (this.bird.y - this.bird.radius < 0) {
            this.bird.y = this.bird.radius;
            this.bird.velocity = 0;
        }
        
        // Particle animations
        this.particles.forEach(p => {
            p.x += p.vx * speed;
            p.y += p.vy * speed;
            p.alpha -= 0.04 * speed;
        });
        this.particles = this.particles.filter(p => p.alpha > 0);
        
        // Spawn pipes
        this.pipeSpawnTimer = (this.pipeSpawnTimer || 0) + speed;
        if (this.pipeSpawnTimer >= 700) {
            this.pipeSpawnTimer -= 700;
            this.spawnPipe();
        }
        
        // Move & collide pipes
        this.pipes.forEach(pipe => {
            pipe.x -= this.pipeSpeed * speed;
            
            // Check pass
            if (!pipe.passed && pipe.x + this.pipeWidth < this.bird.x) {
                pipe.passed = true;
                this.score++;
                this.onScore(this.score);
                playSound('score');
            }
            
            // Collisions
            const inPipeX = (this.bird.x + this.bird.radius > pipe.x && 
                             this.bird.x - this.bird.radius < pipe.x + this.pipeWidth);
                             
            const hitTop = inPipeX && (this.bird.y - this.bird.radius < pipe.topHeight);
            const hitBottom = inPipeX && (this.bird.y + this.bird.radius > this.canvas.height - pipe.bottomHeight);
                               
            if (hitTop || hitBottom) {
                this.gameOver();
            }
        });
        
        this.pipes = this.pipes.filter(pipe => pipe.x + this.pipeWidth > 0);
    }
    
    draw() {
        // Space-y backdrop
        const skyGrad = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        skyGrad.addColorStop(0, '#000000');
        skyGrad.addColorStop(1, '#080808');
        this.ctx.fillStyle = skyGrad;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Background Stars
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
        for (let i = 0; i < 15; i++) {
            const starX = (i * 83 - this.frameCount * 0.3) % this.canvas.width;
            const starY = (i * 47) % (this.canvas.height - 40);
            this.ctx.fillRect(starX < 0 ? starX + this.canvas.width : starX, starY, 2, 2);
        }
        
        // Draw pipes (B&W theme)
        this.pipes.forEach(pipe => {
            const pipeGrad = this.ctx.createLinearGradient(pipe.x, 0, pipe.x + this.pipeWidth, 0);
            pipeGrad.addColorStop(0, '#1c1c1c');
            pipeGrad.addColorStop(1, '#0a0a0a');
            this.ctx.fillStyle = pipeGrad;
            
            this.ctx.shadowBlur = 8;
            this.ctx.shadowColor = 'rgba(255, 255, 255, 0.05)';
            
            // Top pipe body
            this.ctx.fillRect(pipe.x, 0, this.pipeWidth, pipe.topHeight);
            // Top pipe lip
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(pipe.x - 4, pipe.topHeight - 16, this.pipeWidth + 8, 16);
            
            // Bottom pipe body
            this.ctx.fillStyle = pipeGrad;
            this.ctx.fillRect(pipe.x, this.canvas.height - pipe.bottomHeight, this.pipeWidth, pipe.bottomHeight);
            // Bottom pipe lip
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(pipe.x - 4, this.canvas.height - pipe.bottomHeight, this.pipeWidth + 8, 16);
            
            this.ctx.shadowBlur = 0;
        });
        
        // Draw flap particles
        this.particles.forEach(p => {
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.alpha;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1.0;
        
        // Ground bar
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(0, this.canvas.height - 25, this.canvas.width, 25);
        this.ctx.strokeStyle = '#222222';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(0, this.canvas.height - 25);
        this.ctx.lineTo(this.canvas.width, this.canvas.height - 25);
        this.ctx.stroke();
        
        // Bird (Sleek monochrome jet)
        const b = this.bird;
        this.ctx.shadowBlur = 12;
        this.ctx.shadowColor = '#ffffff';
        
        // Body
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.arc(b.x, b.y, b.radius, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
        
        // Eye
        this.ctx.fillStyle = '#000000';
        this.ctx.beginPath();
        this.ctx.arc(b.x + 5, b.y - 3, 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        // Beak
        this.ctx.fillStyle = '#cccccc';
        this.ctx.beginPath();
        this.ctx.moveTo(b.x + b.radius - 2, b.y - 3);
        this.ctx.lineTo(b.x + b.radius + 6, b.y);
        this.ctx.lineTo(b.x + b.radius - 2, b.y + 3);
        this.ctx.closePath();
        this.ctx.fill();
        
        // Wing (wiggles)
        this.ctx.fillStyle = '#888888';
        const wingOffset = b.velocity < 0 ? -3 : 3;
        this.ctx.beginPath();
        this.ctx.ellipse(b.x - 5, b.y + 1, 7, 4 + Math.abs(wingOffset), Math.PI/6, 0, Math.PI * 2);
        this.ctx.fill();
    }
    
    gameOver() {
        this.isPlaying = false;
        playSound('hit');
        this.onGameOver();
    }
}


// ==========================================
// 2. GRID RUNNER GAME CLASS
// ==========================================
class GridRunner {
    constructor(canvas, onScore, onGameOver) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onScore = onScore;
        this.onGameOver = onGameOver;
        
        this.cols = 15;
        this.rows = 10;
        this.cellW = this.canvas.width / this.cols; // 40px
        this.cellH = this.canvas.height / this.rows; // 40px
        
        this.player = { x: 7, y: 5 };
        this.gems = [];
        this.enemies = [];
        
        this.score = 0;
        this.isPlaying = false;
        this.frameCount = 0;
        this.enemyMoveInterval = 200; 
        this.particles = [];
    }
    
    start() {
        this.player = { x: 7, y: 5 };
        this.score = 0;
        this.gems = [];
        this.enemies = [];
        this.isPlaying = true;
        this.frameCount = 0;
        this.enemyMoveInterval = 200;
        this.enemyMoveTimer = 0;
        this.particles = [];
        
        this.spawnGem();
        // Spawn first enemy in corner
        this.enemies.push({ x: 0, y: 0 });
    }
    
    spawnGem() {
        let x, y, occupied;
        do {
            x = Math.floor(Math.random() * this.cols);
            y = Math.floor(Math.random() * this.rows);
            occupied = (x === this.player.x && y === this.player.y) ||
                       this.enemies.some(e => e.x === x && e.y === y) ||
                       this.gems.some(g => g.x === x && g.y === y);
        } while (occupied);
        
        this.gems.push({ x, y, pulse: 0 });
    }
    
    spawnEnemy() {
        let x, y, occupied;
        do {
            x = Math.floor(Math.random() * this.cols);
            y = Math.floor(Math.random() * this.rows);
            const dist = Math.abs(x - this.player.x) + Math.abs(y - this.player.y);
            occupied = dist < 4 ||
                       this.enemies.some(e => e.x === x && e.y === y) ||
                       this.gems.some(g => g.x === x && g.y === y);
        } while (occupied);
        
        this.enemies.push({ x, y });
    }
    
    handleCommand(cmd) {
        if (!this.isPlaying) return;
        
        let dx = 0;
        let dy = 0;
        
        if (cmd === 'UP') dy = -1;
        else if (cmd === 'DOWN') dy = 1;
        else if (cmd === 'YES') dx = 1;  // Yes = Right
        else if (cmd === 'NO') dx = -1;  // No = Left
        
        if (dx !== 0 || dy !== 0) {
            const targetX = this.player.x + dx;
            const targetY = this.player.y + dy;
            
            if (targetX >= 0 && targetX < this.cols && targetY >= 0 && targetY < this.rows) {
                this.player.x = targetX;
                this.player.y = targetY;
                
                // Trail particle effect
                for (let i = 0; i < 5; i++) {
                    this.particles.push({
                        x: (this.player.x - dx + 0.5) * this.cellW,
                        y: (this.player.y - dy + 0.5) * this.cellH,
                        vx: (Math.random() - 0.5) * 3 - dx * 1.5,
                        vy: (Math.random() - 0.5) * 3 - dy * 1.5,
                        alpha: 1.0,
                        size: Math.random() * 2.5 + 1.5,
                        color: 'rgba(255, 255, 255, 0.8)'
                    });
                }
                
                this.checkCollisions();
            }
        }
    }
    
    checkCollisions() {
        // Gem collection
        this.gems.forEach((gem, idx) => {
            if (gem.x === this.player.x && gem.y === this.player.y) {
                this.gems.splice(idx, 1);
                this.score++;
                this.onScore(this.score);
                playSound('score');
                
                // Spark particles
                const px = (gem.x + 0.5) * this.cellW;
                const py = (gem.y + 0.5) * this.cellH;
                for (let i = 0; i < 10; i++) {
                    this.particles.push({
                        x: px,
                        y: py,
                        vx: (Math.random() - 0.5) * 5,
                        vy: (Math.random() - 0.5) * 5,
                        alpha: 1.0,
                        size: Math.random() * 3 + 1,
                        color: '#ffffff'
                    });
                }
                
                this.spawnGem();
                
                // Add difficulty
                if (this.score % 4 === 0) {
                    this.spawnEnemy();
                }
                this.enemyMoveInterval = Math.max(90, 200 - Math.floor(this.score / 2) * 5);
            }
        });
        
        // Enemy check
        if (this.enemies.some(e => e.x === this.player.x && e.y === this.player.y)) {
            this.gameOver();
        }
    }
    
    update() {
        if (!this.isPlaying) return;
        
        const speed = window.gameSpeedMultiplier || 1.0;
        this.frameCount += speed;
        
        // Update particles
        this.particles.forEach(p => {
            p.x += p.vx * speed;
            p.y += p.vy * speed;
            p.alpha -= 0.05 * speed;
        });
        this.particles = this.particles.filter(p => p.alpha > 0);
        
        // Move enemies toward player
        this.enemyMoveTimer = (this.enemyMoveTimer || 0) + speed;
        if (this.enemyMoveTimer >= this.enemyMoveInterval) {
            this.enemyMoveTimer -= this.enemyMoveInterval;
            this.enemies.forEach(e => {
                const dx = this.player.x - e.x;
                const dy = this.player.y - e.y;
                
                if (dx !== 0 && dy !== 0) {
                    if (Math.random() > 0.5) {
                        e.x += Math.sign(dx);
                    } else {
                        e.y += Math.sign(dy);
                    }
                } else if (dx !== 0) {
                    e.x += Math.sign(dx);
                } else if (dy !== 0) {
                    e.y += Math.sign(dy);
                }
            });
            this.checkCollisions();
        }
    }
    
    draw() {
        // Dark grid background
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw gridlines
        this.ctx.strokeStyle = '#121212';
        this.ctx.lineWidth = 1;
        for (let c = 0; c <= this.cols; c++) {
            this.ctx.beginPath();
            this.ctx.moveTo(c * this.cellW, 0);
            this.ctx.lineTo(c * this.cellW, this.canvas.height);
            this.ctx.stroke();
        }
        for (let r = 0; r <= this.rows; r++) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, r * this.cellH);
            this.ctx.lineTo(this.canvas.width, r * this.cellH);
            this.ctx.stroke();
        }
        
        // Draw particles
        this.particles.forEach(p => {
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.alpha;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1.0;
        
        // Draw gems (yellow pulsing diamonds)
        this.gems.forEach(gem => {
            gem.pulse += 0.12;
            const size = 9 + Math.sin(gem.pulse) * 1.5;
            const gx = (gem.x + 0.5) * this.cellW;
            const gy = (gem.y + 0.5) * this.cellH;
            
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = '#ffffff';
            this.ctx.fillStyle = '#ffffff';
            this.ctx.beginPath();
            this.ctx.moveTo(gx, gy - size);
            this.ctx.lineTo(gx + size, gy);
            this.ctx.lineTo(gx, gy + size);
            this.ctx.lineTo(gx - size, gy);
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        });
        
        // Draw enemies (danger spikes)
        this.enemies.forEach(e => {
            const ex = (e.x + 0.5) * this.cellW;
            const ey = (e.y + 0.5) * this.cellH;
            const size = 11;
            
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = '#444444';
            this.ctx.fillStyle = '#222222';
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 1.5;
            
            this.ctx.beginPath();
            this.ctx.moveTo(ex, ey - size);
            this.ctx.lineTo(ex + size, ey + size * 0.7);
            this.ctx.lineTo(ex - size, ey + size * 0.7);
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
            
            // Draw core eye
            this.ctx.fillStyle = '#ffffff';
            this.ctx.beginPath();
            this.ctx.arc(ex, ey + 2, 2.5, 0, Math.PI * 2);
            this.ctx.fill();
            
            this.ctx.shadowBlur = 0;
        });
        
        // Draw player (cyan cyber diamond)
        const px = (this.player.x + 0.5) * this.cellW;
        const py = (this.player.y + 0.5) * this.cellH;
        const size = 11;
        
        this.ctx.shadowBlur = 12;
        this.ctx.shadowColor = '#ffffff';
        this.ctx.fillStyle = '#ffffff';
        
        this.ctx.beginPath();
        this.ctx.moveTo(px, py - size);
        this.ctx.lineTo(px + size, py);
        this.ctx.lineTo(px, py + size);
        this.ctx.lineTo(px - size, py);
        this.ctx.closePath();
        this.ctx.fill();
        
        this.ctx.fillStyle = '#000000';
        this.ctx.beginPath();
        this.ctx.arc(px, py, 3, 0, Math.PI * 2);
        this.ctx.fill();
        
        this.ctx.shadowBlur = 0;
    }
    
    gameOver() {
        this.isPlaying = false;
        playSound('hit');
        this.onGameOver();
    }
}


// ==========================================
// 3. SPACE DEFENDER GAME CLASS
// ==========================================
class SpaceDefender {
    constructor(canvas, onScore, onGameOver) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onScore = onScore;
        this.onGameOver = onGameOver;
        
        this.player = {
            x: 280,
            y: 345,
            w: 40,
            h: 30,
            targetX: 280,
            shieldActive: false,
            shieldTime: 0,
            shieldCooldown: 0
        };
        
        this.lasers = [];
        this.invaders = [];
        this.stars = [];
        this.particles = [];
        
        this.score = 0;
        this.lives = 3;
        this.isPlaying = false;
        this.frameCount = 0;
        this.spawnRate = 350;
        this.lastSpawnFrame = 0;
        
        for (let i = 0; i < 40; i++) {
            this.stars.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                speed: Math.random() * 1.8 + 0.4,
                size: Math.random() * 1.8
            });
        }
    }
    
    start() {
        this.player.x = 280;
        this.player.targetX = 280;
        this.player.shieldActive = false;
        this.player.shieldTime = 0;
        this.player.shieldCooldown = 0;
        
        this.lasers = [];
        this.invaders = [];
        this.particles = [];
        
        this.score = 0;
        this.lives = 3;
        this.isPlaying = true;
        this.frameCount = 0;
        this.spawnRate = 350;
        this.spawnTimer = 0;
    }
    
    handleCommand(cmd) {
        if (!this.isPlaying) return;
        
        if (cmd === 'NO') {
            this.player.targetX = Math.max(20, this.player.targetX - 70);
        } else if (cmd === 'YES') {
            this.player.targetX = Math.min(this.canvas.width - 20 - this.player.w, this.player.targetX + 70);
        } else if (cmd === 'UP') {
            this.lasers.push({
                x: this.player.x + this.player.w / 2,
                y: this.player.y,
                speed: 8.5
            });
            playSound('laser');
        } else if (cmd === 'DOWN') {
            if (this.player.shieldCooldown <= 0) {
                this.player.shieldActive = true;
                this.player.shieldTime = 100; // 1.6s
                this.player.shieldCooldown = 220; // ~3.5s
                playSound('shield');
            }
        }
    }
    
    update() {
        if (!this.isPlaying) return;
        
        const speed = window.gameSpeedMultiplier || 1.0;
        this.frameCount += speed;
        
        // Ship movement smoothing
        this.player.x += (this.player.targetX - this.player.x) * 0.18 * speed;
        
        // Shield state
        if (this.player.shieldActive) {
            this.player.shieldTime -= speed;
            if (this.player.shieldTime <= 0) {
                this.player.shieldActive = false;
            }
        }
        if (this.player.shieldCooldown > 0) {
            this.player.shieldCooldown -= speed;
        }
        
        // Move starfield
        this.stars.forEach(s => {
            s.y += s.speed * speed;
            if (s.y > this.canvas.height) {
                s.y = 0;
                s.x = Math.random() * this.canvas.width;
            }
        });
        
        // Move lasers
        this.lasers.forEach((l, idx) => {
            l.y -= l.speed * speed;
            if (l.y < 0) {
                this.lasers.splice(idx, 1);
            }
        });
        
        // Spawn falling debris / enemies (Max 2 onscreen, 3.5x size)
        this.spawnTimer = (this.spawnTimer || 0) + speed;
        if (this.invaders.length < 2 && this.spawnTimer >= 90) {
            this.spawnTimer = 0;
            const size = (Math.random() * 18 + 14) * 3.5;
            this.invaders.push({
                x: Math.random() * (this.canvas.width - size),
                y: -size,
                w: size,
                h: size,
                speed: Math.random() * 0.15 + 0.12 + (this.score * 0.005),
                rot: Math.random() * Math.PI,
                rotSpeed: (Math.random() - 0.5) * 0.06
            });
        }
        
        // Move invaders
        this.invaders.forEach((inv, invIdx) => {
            inv.y += inv.speed * speed;
            inv.rot += inv.rotSpeed * speed;
            
            // Reached bottom
            if (inv.y > this.canvas.height) {
                this.invaders.splice(invIdx, 1);
                this.loseLife();
            }
            
            // Collide with laser
            this.lasers.forEach((l, lIdx) => {
                const hit = (l.x > inv.x && l.x < inv.x + inv.w &&
                             l.y > inv.y && l.y < inv.y + inv.h);
                if (hit) {
                    this.lasers.splice(lIdx, 1);
                    this.invaders.splice(invIdx, 1);
                    this.score++;
                    this.onScore(this.score);
                    playSound('score');
                    
                    // Explosion spark particles
                    const px = inv.x + inv.w / 2;
                    const py = inv.y + inv.h / 2;
                    for (let i = 0; i < 12; i++) {
                        this.particles.push({
                            x: px,
                            y: py,
                            vx: (Math.random() - 0.5) * 7,
                            vy: (Math.random() - 0.5) * 7,
                            alpha: 1.0,
                            size: Math.random() * 3 + 1,
                            color: '#ffffff'
                        });
                    }
                }
            });
            
            // Collide with ship
            const p = this.player;
            const shipHit = (inv.x + inv.w > p.x && inv.x < p.x + p.w &&
                             inv.y + inv.h > p.y && inv.y < p.y + p.h);
            if (shipHit) {
                this.invaders.splice(invIdx, 1);
                if (p.shieldActive) {
                    playSound('shield_blocked');
                    const px = p.x + p.w / 2;
                    const py = p.y + p.h / 2;
                    for (let i = 0; i < 15; i++) {
                        this.particles.push({
                            x: px,
                            y: py,
                            vx: (Math.random() - 0.5) * 5,
                            vy: (Math.random() - 0.5) * 5,
                            alpha: 1.0,
                            size: Math.random() * 2 + 1,
                            color: '#ffffff'
                        });
                    }
                } else {
                    playSound('hit');
                    this.loseLife();
                }
            }
        });
        
        // Particle fade
        this.particles.forEach(p => {
            p.x += p.vx * speed;
            p.y += p.vy * speed;
            p.alpha -= 0.035 * speed;
        });
        this.particles = this.particles.filter(p => p.alpha > 0);
    }
    
    loseLife() {
        if (!this.isPlaying) return;
        this.lives--;
        if (this.lives <= 0) {
            this.gameOver();
        }
    }
    
    draw() {
        // Cosmos theme (monochrome)
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Stars
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        this.stars.forEach(s => {
            this.ctx.fillRect(s.x, s.y, s.size, s.size);
        });
        
        // Draw lasers (white beams)
        this.lasers.forEach(l => {
            this.ctx.shadowBlur = 6;
            this.ctx.shadowColor = '#ffffff';
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillRect(l.x - 2, l.y, 4, 12);
            this.ctx.shadowBlur = 0;
        });
        
        // Draw debris (rocks - styled in B&W outline)
        this.invaders.forEach(inv => {
            this.ctx.save();
            const cx = inv.x + inv.w / 2;
            const cy = inv.y + inv.h / 2;
            this.ctx.translate(cx, cy);
            this.ctx.rotate(inv.rot);
            
            this.ctx.shadowBlur = 6;
            this.ctx.shadowColor = '#333333';
            this.ctx.fillStyle = '#1c1c1c';
            this.ctx.strokeStyle = '#cccccc';
            this.ctx.lineWidth = 2;
            
            this.ctx.beginPath();
            const points = 7;
            for (let i = 0; i < points; i++) {
                const angle = (i / points) * Math.PI * 2;
                const dist = (inv.w / 2) * (0.85 + Math.sin(i * 1.8) * 0.12);
                const px = Math.cos(angle) * dist;
                const py = Math.sin(angle) * dist;
                if (i === 0) this.ctx.moveTo(px, py);
                else this.ctx.lineTo(px, py);
            }
            this.ctx.closePath();
            this.ctx.fill();
            this.ctx.stroke();
            
            this.ctx.restore();
            this.ctx.shadowBlur = 0;
        });
        
        // Draw particles
        this.particles.forEach(p => {
            this.ctx.fillStyle = p.color;
            this.ctx.globalAlpha = p.alpha;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            this.ctx.fill();
        });
        this.ctx.globalAlpha = 1.0;
        
        // Draw player ship
        const p = this.player;
        
        if (p.shieldActive) {
            this.ctx.save();
            this.ctx.globalAlpha = 0.22 + (Math.sin(this.frameCount * 0.2) * 0.08);
            this.ctx.shadowBlur = 12;
            this.ctx.shadowColor = '#ffffff';
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
            this.ctx.beginPath();
            this.ctx.arc(p.x + p.w / 2, p.y + p.h / 2, Math.max(p.w, p.h) * 0.9, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.lineWidth = 1.5;
            this.ctx.stroke();
            this.ctx.restore();
            this.ctx.shadowBlur = 0;
        }
        
        // Spaceship shape
        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.moveTo(p.x + p.w / 2, p.y); // nose
        this.ctx.lineTo(p.x + p.w, p.y + p.h); // wing right
        this.ctx.lineTo(p.x + p.w * 0.75, p.y + p.h * 0.8);
        this.ctx.lineTo(p.x + p.w * 0.25, p.y + p.h * 0.8);
        this.ctx.lineTo(p.x, p.y + p.h); // wing left
        this.ctx.closePath();
        this.ctx.fill();
        
        // Cockpit
        this.ctx.fillStyle = '#000000';
        this.ctx.beginPath();
        this.ctx.moveTo(p.x + p.w / 2, p.y + 6);
        this.ctx.lineTo(p.x + p.w * 0.62, p.y + p.h * 0.52);
        this.ctx.lineTo(p.x + p.w * 0.38, p.y + p.h * 0.52);
        this.ctx.closePath();
        this.ctx.fill();
        
        // Engine fire (gray thruster glow)
        if (this.frameCount % 4 < 2) {
            this.ctx.fillStyle = '#888888';
            this.ctx.beginPath();
            this.ctx.moveTo(p.x + p.w * 0.38, p.y + p.h * 0.82);
            this.ctx.lineTo(p.x + p.w / 2, p.y + p.h * 1.15);
            this.ctx.lineTo(p.x + p.w * 0.62, p.y + p.h * 0.82);
            this.ctx.closePath();
            this.ctx.fill();
        }
        
        // Draw HUD overlay (Lives and Shield Cooldown)
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '600 12px "Outfit", sans-serif';
        this.ctx.fillText('LIVES: ', 15, 22);
        for (let i = 0; i < this.lives; i++) {
            // Draw a neat vector heart
            const hx = 60 + i * 18;
            const hy = 12;
            const hs = 10;
            this.ctx.beginPath();
            this.ctx.moveTo(hx, hy + hs / 4);
            this.ctx.bezierCurveTo(hx, hy, hx - hs/2, hy, hx - hs/2, hy + hs/2);
            this.ctx.bezierCurveTo(hx - hs/2, hy + hs*0.8, hx, hy + hs*1.1, hx, hy + hs);
            this.ctx.bezierCurveTo(hx, hy + hs*1.1, hx + hs/2, hy + hs*0.8, hx + hs/2, hy + hs/2);
            this.ctx.bezierCurveTo(hx + hs/2, hy, hx, hy, hx, hy + hs / 4);
            this.ctx.fill();
        }
        
        // Shield status
        if (p.shieldCooldown > 0) {
            this.ctx.fillStyle = '#888888';
            this.ctx.fillText(`SHIELD COOLDOWN: ${(p.shieldCooldown / 60).toFixed(1)}s`, this.canvas.width - 180, 22);
        } else {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fillText('SHIELD READY [DOWN]', this.canvas.width - 180, 22);
        }
    }
    
    gameOver() {
        this.isPlaying = false;
        playSound('hit');
        this.onGameOver();
    }
}

// ==========================================
// 4. SIMON MEMORY GAME CLASS (TURN-BASED)
// ==========================================
class SimonMemory {
    constructor(canvas, onScore, onGameOver) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onScore = onScore;
        this.onGameOver = onGameOver;
        
        this.sequence = [];
        this.playerSequence = [];
        this.state = 'showing'; // 'showing', 'inputting', 'success', 'gameover'
        this.showIndex = 0;
        this.showTimer = 0;
        this.message = "Watch the sequence!";
        this.score = 0;
        this.isPlaying = false;
        
        // Colors for Simon buttons (B&W/Grayscale)
        this.panels = {
            'UP': { color: 'rgba(255, 255, 255, 0.05)', activeColor: 'rgba(255, 255, 255, 0.95)', x: 230, y: 50, w: 140, h: 80 },
            'DOWN': { color: 'rgba(255, 255, 255, 0.05)', activeColor: 'rgba(255, 255, 255, 0.95)', x: 230, y: 250, w: 140, h: 80 },
            'YES': { color: 'rgba(255, 255, 255, 0.05)', activeColor: 'rgba(255, 255, 255, 0.95)', x: 390, y: 150, w: 140, h: 80 },
            'NO': { color: 'rgba(255, 255, 255, 0.05)', activeColor: 'rgba(255, 255, 255, 0.95)', x: 70, y: 150, w: 140, h: 80 }
        };
        this.activePanel = null;
    }
    
    start() {
        this.score = 0;
        this.sequence = [];
        this.isPlaying = true;
        this.nextRound();
    }
    
    nextRound() {
        const words = ['UP', 'DOWN', 'YES', 'NO'];
        const randomWord = words[Math.floor(Math.random() * words.length)];
        this.sequence.push(randomWord);
        
        this.playerSequence = [];
        this.state = 'showing';
        this.showIndex = 0;
        this.showTimer = 35; // Initial delay before sequence starts
        this.activePanel = null;
        this.message = "Watch the sequence!";
    }
    
    handleCommand(cmd) {
        if (!this.isPlaying || this.state !== 'inputting') return;
        
        const validWords = ['UP', 'DOWN', 'YES', 'NO'];
        if (!validWords.includes(cmd)) return;
        
        this.playerSequence.push(cmd);
        
        // Flash the clicked panel
        this.activePanel = cmd;
        this.showTimer = 15; // Flash for 15 frames
        
        const stepIdx = this.playerSequence.length - 1;
        if (cmd === this.sequence[stepIdx]) {
            // Correct input
            if (this.playerSequence.length === this.sequence.length) {
                // Sequence completed!
                this.score++;
                this.onScore(this.score);
                this.state = 'success';
                this.showTimer = 40; // Success pause
                this.message = "Perfect! Keep it up!";
                playSound('score');
            } else {
                playSound('score'); // Simple correct feedback sound
            }
        } else {
            // Wrong sequence
            this.gameOver();
        }
    }
    
    update() {
        if (!this.isPlaying) return;
        
        const speed = window.gameSpeedMultiplier || 1.0;
        
        if (this.state === 'showing') {
            if (this.showTimer > 0) {
                this.showTimer -= speed;
            }
            if (this.showTimer <= 0) {
                if (this.activePanel !== null) {
                    // We were flashing a panel, now turn it off and pause
                    this.activePanel = null;
                    this.showIndex++;
                    if (this.showIndex >= this.sequence.length) {
                        this.state = 'inputting';
                        this.message = "Your turn! Repeat the sequence.";
                    } else {
                        this.showTimer = 15; // Pause between flashes
                    }
                } else {
                    // We were in initial delay or inter-panel pause, now flash the next panel
                    this.activePanel = this.sequence[this.showIndex];
                    this.showTimer = 25; // Glow duration
                    playSound('flap'); // Play light blip
                }
            }
        } else {
            if (this.showTimer > 0) {
                this.showTimer -= speed;
                if (this.showTimer <= 0) {
                    this.activePanel = null;
                    if (this.state === 'success') {
                        this.nextRound();
                    }
                }
            }
        }
    }
    
    draw() {
        // Dark theme background
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Grid background lines for cyber look
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.02)';
        this.ctx.lineWidth = 1;
        for (let i = 0; i < this.canvas.width; i += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(i, 0);
            this.ctx.lineTo(i, this.canvas.height);
            this.ctx.stroke();
        }
        for (let i = 0; i < this.canvas.height; i += 40) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, i);
            this.ctx.lineTo(this.canvas.width, i);
            this.ctx.stroke();
        }
        
        // Draw Header
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '700 16px "Outfit", sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('SIMON MEMORY', this.canvas.width / 2, 28);
        
        // Draw instruction message
        this.ctx.fillStyle = this.state === 'inputting' ? '#ffffff' : '#888888';
        this.ctx.font = '500 13px "Outfit", sans-serif';
        this.ctx.fillText(this.message, this.canvas.width / 2, 380);
        
        // Draw panels
        for (let key in this.panels) {
            const p = this.panels[key];
            const isActive = this.activePanel === key;
            
            this.ctx.save();
            if (isActive) {
                this.ctx.shadowBlur = 15;
                this.ctx.shadowColor = p.activeColor;
                this.ctx.fillStyle = p.activeColor;
            } else {
                this.ctx.fillStyle = p.color;
            }
            
            // Draw panel box
            this.ctx.strokeStyle = isActive ? '#ffffff' : 'rgba(255,255,255,0.08)';
            this.ctx.lineWidth = isActive ? 2 : 1;
            
            drawRoundRect(this.ctx, p.x, p.y, p.w, p.h, 8);
            this.ctx.fill();
            this.ctx.stroke();
            
            // Label text inside panel
            this.ctx.shadowBlur = 0;
            this.ctx.fillStyle = isActive ? '#000000' : '#ffffff';
            this.ctx.font = '700 14px "Outfit", sans-serif';
            this.ctx.fillText(key, p.x + p.w / 2, p.y + p.h / 2 + 5);
            this.ctx.restore();
        }
        
        // Draw progress status
        if (this.state === 'inputting') {
            this.ctx.fillStyle = '#888888';
            this.ctx.font = '500 12px "Outfit", sans-serif';
            this.ctx.fillText(`Progress: ${this.playerSequence.length} / ${this.sequence.length}`, this.canvas.width / 2, 220);
        }
    }
    
    gameOver() {
        this.isPlaying = false;
        playSound('hit');
        this.onGameOver();
    }
}

// ==========================================
// 5. CYBER HI-LO GAME CLASS (TURN-BASED)
// ==========================================
class CyberHiLo {
    constructor(canvas, onScore, onGameOver) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.onScore = onScore;
        this.onGameOver = onGameOver;
        
        this.currentCard = 0;
        this.nextCard = 0;
        this.guess = null; // 'UP' (Higher) or 'DOWN' (Lower)
        this.message = "Is the next card Higher or Lower?";
        this.score = 0;
        this.lives = 3;
        this.isPlaying = false;
        this.state = 'waiting'; // 'waiting', 'revealing'
        this.revealTimer = 0;
    }
    
    start() {
        this.score = 0;
        this.lives = 3;
        this.isPlaying = true;
        this.state = 'waiting';
        this.currentCard = Math.floor(Math.random() * 13) + 1;
        this.nextCard = 0;
        this.guess = null;
        this.message = "Say UP for Higher, DOWN for Lower!";
    }
    
    cardName(val) {
        if (val === 1) return 'Ace';
        if (val === 11) return 'Jack';
        if (val === 12) return 'Queen';
        if (val === 13) return 'King';
        return val.toString();
    }
    
    handleCommand(cmd) {
        if (!this.isPlaying || this.state !== 'waiting') return;
        if (cmd !== 'UP' && cmd !== 'DOWN') return;
        
        this.guess = cmd;
        
        // Choose next card
        do {
            this.nextCard = Math.floor(Math.random() * 13) + 1;
        } while (this.nextCard === this.currentCard);
        
        const isUp = this.nextCard > this.currentCard;
        const correct = (cmd === 'UP' && isUp) || (cmd === 'DOWN' && !isUp);
        
        this.state = 'revealing';
        this.revealTimer = 100; // 1.6s showing result
        
        if (correct) {
            this.score++;
            this.onScore(this.score);
            this.message = `Correct! It's a ${this.cardName(this.nextCard)}.`;
            playSound('score');
        } else {
            this.lives--;
            this.message = `Wrong! It's a ${this.cardName(this.nextCard)}.`;
            playSound('hit');
        }
    }
    
    update() {
        if (!this.isPlaying) return;
        
        const speed = window.gameSpeedMultiplier || 1.0;
        
        if (this.state === 'revealing') {
            this.revealTimer -= speed;
            if (this.revealTimer <= 0) {
                if (this.lives <= 0) {
                    this.gameOver();
                } else {
                    this.currentCard = this.nextCard;
                    this.nextCard = 0;
                    this.guess = null;
                    this.state = 'waiting';
                    this.message = "Say UP for Higher, DOWN for Lower!";
                }
            }
        }
    }
    
    draw() {
        // Pure black casino table background
        this.ctx.fillStyle = '#000000';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Gray borders
        this.ctx.strokeStyle = '#333333';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(10, 10, this.canvas.width - 20, this.canvas.height - 20);
        
        // Draw Header
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '700 16px "Outfit", sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('CYBER HI-LO', this.canvas.width / 2, 32);
        
        // Draw Hearts/Lives as vector graphics centered
        this.ctx.fillStyle = '#ffffff';
        for (let i = 0; i < 3; i++) {
            const hx = this.canvas.width / 2 - (3 * 18) / 2 + i * 18;
            const hy = 48;
            const hs = 10;
            this.ctx.beginPath();
            this.ctx.moveTo(hx, hy + hs / 4);
            this.ctx.bezierCurveTo(hx, hy, hx - hs/2, hy, hx - hs/2, hy + hs/2);
            this.ctx.bezierCurveTo(hx - hs/2, hy + hs*0.8, hx, hy + hs*1.1, hx, hy + hs);
            this.ctx.bezierCurveTo(hx, hy + hs*1.1, hx + hs/2, hy + hs*0.8, hx + hs/2, hy + hs/2);
            this.ctx.bezierCurveTo(hx + hs/2, hy, hx, hy, hx, hy + hs / 4);
            
            if (i < this.lives) {
                this.ctx.fillStyle = '#ffffff';
                this.ctx.fill();
            } else {
                this.ctx.strokeStyle = '#222222';
                this.ctx.lineWidth = 1.5;
                this.ctx.stroke();
            }
        }
        
        // Draw Card slots
        const cardW = 100;
        const cardH = 150;
        const cardY = 110;
        
        // Card 1: Current Card (Left)
        this.drawCard(150, cardY, cardW, cardH, this.currentCard, false);
        
        // Card 2: Next Card (Right)
        if (this.state === 'revealing') {
            this.drawCard(350, cardY, cardW, cardH, this.nextCard, false);
        } else {
            this.drawCard(350, cardY, cardW, cardH, 0, true); // face down
        }
        
        // Labels
        this.ctx.fillStyle = '#888888';
        this.ctx.font = '600 11px "Outfit", sans-serif';
        this.ctx.fillText('CURRENT CARD', 200, 280);
        this.ctx.fillText('NEXT CARD', 400, 280);
        
        // Guess Arrow indicator
        if (this.guess) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '700 24px "Outfit", sans-serif';
            this.ctx.fillText(this.guess === 'UP' ? '▲ HIGHER' : '▼ LOWER', this.canvas.width / 2, 320);
        }
        
        // Draw Instruction / Message
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = '600 13px "Outfit", sans-serif';
        this.ctx.fillText(this.message, this.canvas.width / 2, 355);
    }
    
    drawCard(x, y, w, h, val, faceDown) {
        this.ctx.save();
        
        // Shadow card background
        this.ctx.shadowBlur = 10;
        this.ctx.shadowColor = 'rgba(255, 255, 255, 0.1)';
        
        // Draw card base
        this.ctx.fillStyle = faceDown ? '#121212' : '#ffffff';
        this.ctx.strokeStyle = faceDown ? '#444444' : '#e2e8f0';
        this.ctx.lineWidth = 2;
        
        drawRoundRect(this.ctx, x, y, w, h, 8);
        this.ctx.fill();
        this.ctx.stroke();
        this.ctx.shadowBlur = 0;
        
        if (faceDown) {
            // Draw card back details (monochrome grid + question mark)
            this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
            this.ctx.lineWidth = 1;
            for (let i = x + 10; i < x + w; i += 12) {
                this.ctx.beginPath();
                this.ctx.moveTo(i, y);
                this.ctx.lineTo(i, y + h);
                this.ctx.stroke();
            }
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '700 28px "Outfit", sans-serif';
            this.ctx.fillText('?', x + w/2, y + h/2 + 10);
        } else {
            // Draw card face details (numbers and suits - all black-and-white)
            const isRed = val % 2 === 0;
            this.ctx.fillStyle = '#0f172a'; // force all text to dark charcoal for monochrome
            this.ctx.font = '700 20px "Outfit", sans-serif';
            this.ctx.textAlign = 'left';
            
            // Top-left text
            const displayVal = val === 1 ? 'A' : val === 11 ? 'J' : val === 12 ? 'Q' : val === 13 ? 'K' : val.toString();
            this.ctx.fillText(displayVal, x + 8, y + 24);
            
            // Suit Center
            const suit = isRed ? '✦' : '⚛';
            this.ctx.font = '36px "Outfit", sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(suit, x + w / 2, y + h / 2 + 12);
            
            // Bottom-right text
            this.ctx.font = '700 20px "Outfit", sans-serif';
            this.ctx.textAlign = 'right';
            this.ctx.fillText(displayVal, x + w - 8, y + h - 12);
        }
        
        this.ctx.restore();
    }
    
    gameOver() {
        this.isPlaying = false;
        playSound('hit');
        this.onGameOver();
    }
}

// ==========================================
// GAME MANAGER MODULE
// ==========================================
let activeGame = null;
let activeGameName = 'flappy';
let animationId = null;

const highscores = {
    flappy: 0,
    grid: 0,
    shooter: 0,
    simon: 0,
    hilo: 0
};

let uiElements = {};

export function initGames(canvas, overlay, startBtn, scoreEl, highscoreEl, lastCommandEl) {
    uiElements = { canvas, overlay, startBtn, scoreEl, highscoreEl, lastCommandEl };
    
    // Set default highscores from localStorage
    try {
        highscores.flappy = parseInt(localStorage.getItem('highscore_flappy') || '0', 10);
        highscores.grid = parseInt(localStorage.getItem('highscore_grid') || '0', 10);
        highscores.shooter = parseInt(localStorage.getItem('highscore_shooter') || '0', 10);
        highscores.simon = parseInt(localStorage.getItem('highscore_simon') || '0', 10);
        highscores.hilo = parseInt(localStorage.getItem('highscore_hilo') || '0', 10);
    } catch (e) {
        console.warn("Storage access failed:", e);
    }
    
    updateHighScoreDisplay();
    
    // Keyboard Event Listener
    window.addEventListener('keydown', (e) => {
        if (!activeGame || !activeGame.isPlaying) return;
        
        let cmd = null;
        if (e.code === 'ArrowUp' || e.code === 'Space') {
            cmd = 'UP';
            e.preventDefault();
        } else if (e.code === 'ArrowDown') {
            cmd = 'DOWN';
            e.preventDefault();
        } else if (e.code === 'ArrowRight' || e.code === 'KeyD') {
            cmd = 'YES'; // Right
            e.preventDefault();
        } else if (e.code === 'ArrowLeft' || e.code === 'KeyA') {
            cmd = 'NO'; // Left
            e.preventDefault();
        }
        
        if (cmd) {
            handleGameVoiceCommand(cmd);
        }
    });
    
    // Init Game
    switchGame('flappy');
}

export function switchGame(gameName) {
    stopActiveGame();
    activeGameName = gameName;
    
    const ctx = uiElements.canvas.getContext('2d');
    ctx.clearRect(0, 0, uiElements.canvas.width, uiElements.canvas.height);
    
    if (uiElements.lastCommandEl) {
        uiElements.lastCommandEl.textContent = 'Waiting...';
        uiElements.lastCommandEl.className = 'metric-value empty';
    }
    
    // Set title and details on overlay
    const titleEl = document.getElementById('overlay-title');
    const descEl = document.getElementById('overlay-instructions');
    
    if (gameName === 'flappy') {
        titleEl.textContent = 'Flappy Bird';
        descEl.innerHTML = 'Control the bird by saying <b>"UP"</b> to flap!<br>Avoid the obstacles to score points.<br><span class="font-small text-muted">(Keyboard: Space / ArrowUp)</span>';
        activeGame = new FlappyBird(uiElements.canvas, onScore, onGameOver);
    } else if (gameName === 'grid') {
        titleEl.textContent = 'Grid Runner';
        descEl.innerHTML = 'Collect glowing stars by moving around the grid!<br>Controls:<br>Say <b>"UP"</b> / <b>"DOWN"</b> to move vertically.<br>Say <b>"YES"</b> (Right) / <b>"NO"</b> (Left) to move horizontally.<br><span class="font-small text-muted">(Keyboard: W,A,S,D / Arrow Keys)</span>';
        activeGame = new GridRunner(uiElements.canvas, onScore, onGameOver);
    } else if (gameName === 'shooter') {
        titleEl.textContent = 'Space Defender';
        descEl.innerHTML = 'Defend space from falling hazards!<br>Controls:<br>Say <b>"NO"</b> (Left) / <b>"YES"</b> (Right) to slide ship.<br>Say <b>"UP"</b> to shoot lasers.<br>Say <b>"DOWN"</b> to activate energy shield (3s cooldown).<br><span class="font-small text-muted">(Keyboard: Left/Right to move, Up to shoot, Down for shield)</span>';
        activeGame = new SpaceDefender(uiElements.canvas, onScore, onGameOver);
    } else if (gameName === 'simon') {
        titleEl.textContent = 'Simon Memory';
        descEl.innerHTML = 'Fully Turn-Based!<br>Repeat the sequence flashed by the system.<br>No time pressure. Wait for each keyword detection.<br><span class="font-small text-muted">(Keyboard: W,A,S,D / Arrow Keys)</span>';
        activeGame = new SimonMemory(uiElements.canvas, onScore, onGameOver);
    } else if (gameName === 'hilo') {
        titleEl.textContent = 'Cyber Hi-Lo';
        descEl.innerHTML = 'Fully Turn-Based!<br>Guess if the next card is higher or lower than the current card.<br>Say <b>"UP"</b> for Higher, <b>"DOWN"</b> for Lower.<br><span class="font-small text-muted">(Keyboard: ArrowUp / ArrowDown)</span>';
        activeGame = new CyberHiLo(uiElements.canvas, onScore, onGameOver);
    }
    
    uiElements.overlay.classList.add('visible');
    uiElements.scoreEl.textContent = '0';
    updateHighScoreDisplay();
}

function onScore(score) {
    uiElements.scoreEl.textContent = score;
    if (score > highscores[activeGameName]) {
        highscores[activeGameName] = score;
        updateHighScoreDisplay();
        try {
            localStorage.setItem(`highscore_${activeGameName}`, score.toString());
        } catch (e) {}
    }
}

function onGameOver() {
    uiElements.overlay.classList.add('visible');
    const titleEl = document.getElementById('overlay-title');
    titleEl.textContent = 'Game Over';
    
    const startBtn = document.getElementById('start-game-btn');
    startBtn.textContent = 'Try Again';
}

function updateHighScoreDisplay() {
    if (uiElements.highscoreEl) {
        uiElements.highscoreEl.textContent = highscores[activeGameName];
    }
}

export function startGame() {
    if (!activeGame) return;
    
    uiElements.overlay.classList.remove('visible');
    uiElements.scoreEl.textContent = '0';
    
    if (uiElements.lastCommandEl) {
        uiElements.lastCommandEl.textContent = 'Waiting...';
        uiElements.lastCommandEl.className = 'metric-value empty';
    }
    
    activeGame.start();
    
    // Start game tick
    if (animationId) cancelAnimationFrame(animationId);
    
    function tick() {
        if (activeGame && activeGame.isPlaying) {
            activeGame.update();
            activeGame.draw();
            animationId = requestAnimationFrame(tick);
        }
    }
    tick();
}

export function stopActiveGame() {
    if (activeGame) {
        activeGame.isPlaying = false;
    }
    if (animationId) {
        cancelAnimationFrame(animationId);
        animationId = null;
    }
}

export function handleGameVoiceCommand(command) {
    if (!activeGame || !activeGame.isPlaying) return;
    
    // Flappy bird only up keyword
    if (activeGameName === 'flappy' && command !== 'UP') {
        return; 
    }
    
    // Display command visually
    if (uiElements.lastCommandEl) {
        uiElements.lastCommandEl.textContent = command;
        
        // Remove previous command styling classes
        uiElements.lastCommandEl.classList.remove('cmd-yes', 'cmd-no', 'cmd-up', 'cmd-down', 'cmd-other', 'empty');
        
        // Add specific color class
        const cmdLower = command.toLowerCase();
        if (['yes', 'no', 'up', 'down', 'other'].includes(cmdLower)) {
            uiElements.lastCommandEl.classList.add(`cmd-${cmdLower}`);
        } else {
            uiElements.lastCommandEl.classList.add('cmd-other');
        }
        
        uiElements.lastCommandEl.classList.add('highlight-flash');
        setTimeout(() => {
            uiElements.lastCommandEl.classList.remove('highlight-flash');
        }, 300);
    }
    
    activeGame.handleCommand(command);
}
