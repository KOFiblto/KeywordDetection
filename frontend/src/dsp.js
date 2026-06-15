// DSP implementation matching PyTorch/torchaudio MFCC & MelSpectrogram exactly.

let twiddleCache = {};
function getTwiddleFactors(nFft) {
    if (twiddleCache[nFft]) {
        return twiddleCache[nFft];
    }
    const halfN = Math.floor(nFft / 2) + 1;
    const cosTable = new Float32Array(halfN * nFft);
    const sinTable = new Float32Array(halfN * nFft);
    const angleStep = (2 * Math.PI) / nFft;
    
    for (let k = 0; k < halfN; k++) {
        const angleK = k * angleStep;
        const offset = k * nFft;
        for (let n = 0; n < nFft; n++) {
            const angle = n * angleK;
            cosTable[offset + n] = Math.cos(angle);
            sinTable[offset + n] = Math.sin(angle);
        }
    }
    
    const factors = { cosTable, sinTable };
    twiddleCache[nFft] = factors;
    return factors;
}

let windowCache = {};
function getWindow(nFft) {
    if (windowCache[nFft]) {
        return windowCache[nFft];
    }
    const window = new Float32Array(nFft);
    for (let i = 0; i < nFft; i++) {
        window[i] = Math.pow(Math.sin((Math.PI * i) / nFft), 2);
    }
    windowCache[nFft] = window;
    return window;
}

let dctCache = {};
function getDctTable(rows) {
    if (dctCache[rows]) {
        return dctCache[rows];
    }
    const table = new Float32Array(rows * rows);
    for (let k = 0; k < rows; k++) {
        const angleScale = (Math.PI * k) / (2.0 * rows);
        const offset = k * rows;
        for (let n = 0; n < rows; n++) {
            table[offset + n] = Math.cos(angleScale * (2 * n + 1));
        }
    }
    dctCache[rows] = table;
    return table;
}

let filterbankCache = {};
function getMelFilterbank(nFreqs, fMin, fMax, nMels, sampleRate, norm = null, melScale = 'htk') {
    const key = `${nFreqs}_${fMin}_${fMax}_${nMels}_${sampleRate}_${norm}_${melScale}`;
    if (filterbankCache[key]) {
        return filterbankCache[key];
    }
    const fb = createMelFilterbank(nFreqs, fMin, fMax, nMels, sampleRate, norm, melScale);
    filterbankCache[key] = fb;
    return fb;
}

function hzToMel(freq, melScale = 'htk') {
    if (melScale === 'htk') {
        return 2595.0 * Math.log10(1.0 + (freq / 700.0));
    }
    if (freq < 1000.0) {
        return freq * 3.0 / 200.0;
    }
    return 15.0 + Math.log(freq / 1000.0) / (Math.log(6.4) / 27.0);
}

function melToHz(mels, melScale = 'htk') {
    if (melScale === 'htk') {
        return 700.0 * (Math.pow(10.0, mels / 2595.0) - 1.0);
    }
    // Slaney scale
    let freq = mels * 200.0 / 3.0;
    const minLogMel = 15.0;
    if (mels >= minLogMel) {
        freq = 1000.0 * Math.exp((mels - minLogMel) * (Math.log(6.4) / 27.0));
    }
    return freq;
}

function createMelFilterbank(nFreqs, fMin, fMax, nMels, sampleRate, norm = null, melScale = 'htk') {
    const allFreqs = new Float32Array(nFreqs);
    for (let i = 0; i < nFreqs; i++) {
        allFreqs[i] = (i * (sampleRate / 2.0)) / (nFreqs - 1);
    }

    const mMin = hzToMel(fMin, melScale);
    const mMax = hzToMel(fMax, melScale);

    const mPts = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
        mPts[i] = mMin + (i * (mMax - mMin)) / (nMels + 1);
    }

    const fPts = new Float32Array(nMels + 2);
    for (let i = 0; i < nMels + 2; i++) {
        fPts[i] = melToHz(mPts[i], melScale);
    }

    const fb = new Float32Array(nFreqs * nMels);
    const fDiff = new Float32Array(nMels + 1);
    for (let i = 0; i < nMels + 1; i++) {
        fDiff[i] = fPts[i + 1] - fPts[i];
    }

    for (let i = 0; i < nFreqs; i++) {
        const f = allFreqs[i];
        for (let j = 0; j < nMels; j++) {
            const downSlope = -(fPts[j] - f) / fDiff[j];
            const upSlope = (fPts[j + 2] - f) / fDiff[j + 1];
            
            let val = Math.max(0.0, Math.min(downSlope, upSlope));
            fb[i * nMels + j] = val;
        }
    }

    if (norm === 'slaney') {
        for (let j = 0; j < nMels; j++) {
            const enorm = 2.0 / (fPts[j + 2] - fPts[j]);
            for (let i = 0; i < nFreqs; i++) {
                fb[i * nMels + j] *= enorm;
            }
        }
    }

    return fb;
}

function dftPowerInPlace(frame, nFft, spectrogram, f, nFrames) {
    const halfN = Math.floor(nFft / 2) + 1;
    const { cosTable, sinTable } = getTwiddleFactors(nFft);
    
    for (let k = 0; k < halfN; k++) {
        let sumReal = 0;
        let sumImag = 0;
        const offset = k * nFft;
        for (let n = 0; n < frame.length; n++) {
            const val = frame[n];
            sumReal += val * cosTable[offset + n];
            sumImag -= val * sinTable[offset + n];
        }
        spectrogram[k * nFrames + f] = sumReal * sumReal + sumImag * sumImag;
    }
}

function stft(x, nFft, hopLength) {
    const window = getWindow(nFft);

    const padLen = Math.floor(nFft / 2);
    const padded = new Float32Array(x.length + 2 * padLen);
    
    for (let i = 0; i < padLen; i++) {
        padded[i] = x[padLen - i];
    }
    for (let i = 0; i < x.length; i++) {
        padded[padLen + i] = x[i];
    }
    const lastIdx = x.length - 1;
    for (let i = 0; i < padLen; i++) {
        padded[padLen + x.length + i] = x[lastIdx - 1 - i];
    }

    const nFrames = 1 + Math.floor((padded.length - nFft) / hopLength);
    const halfN = Math.floor(nFft / 2) + 1;
    const spectrogram = new Float32Array(halfN * nFrames);

    const frame = new Float32Array(nFft);
    for (let f = 0; f < nFrames; f++) {
        const start = f * hopLength;
        for (let i = 0; i < nFft; i++) {
            frame[i] = padded[start + i] * window[i];
        }
        dftPowerInPlace(frame, nFft, spectrogram, f, nFrames);
    }

    return { data: spectrogram, rows: halfN, cols: nFrames };
}

function computeDct2(melSpec, rows, cols) {
    const dct = new Float32Array(rows * cols);
    const factor0 = 0.5 * Math.sqrt(1.0 / rows);
    const factor1 = 0.5 * Math.sqrt(2.0 / rows);
    const dctTable = getDctTable(rows);

    for (let k = 0; k < rows; k++) {
        const scale = (k === 0) ? factor0 : factor1;
        const offset = k * rows;
        
        for (let c = 0; c < cols; c++) {
            let sum = 0;
            for (let n = 0; n < rows; n++) {
                sum += melSpec[n * cols + c] * dctTable[offset + n];
            }
            dct[k * cols + c] = 2.0 * sum * scale;
        }
    }
    return dct;
}

export function preprocessMelSpectrogram(x, sampleRate, nFft = 400, hopLength = 160, nMels = 64) {
    const { data: power, rows: nFreqs, cols: nFrames } = stft(x, nFft, hopLength);
    const fb = getMelFilterbank(nFreqs, 0.0, sampleRate / 2.0, nMels, sampleRate);

    const mel = new Float32Array(nMels * nFrames);
    for (let m = 0; m < nMels; m++) {
        for (let f = 0; f < nFrames; f++) {
            let sum = 0;
            for (let k = 0; k < nFreqs; k++) {
                sum += fb[k * nMels + m] * power[k * nFrames + f];
            }
            mel[m * nFrames + f] = Math.log(sum + 1e-6);
        }
    }
    return { data: mel, rows: nMels, cols: nFrames };
}

export function preprocessMfcc(x, sampleRate, nMfcc = 40, nMels = 64) {
    const nFft = 400;
    const hopLength = 200;
    const { data: power, rows: nFreqs, cols: nFrames } = stft(x, nFft, hopLength);
    const fb = getMelFilterbank(nFreqs, 0.0, sampleRate / 2.0, nMels, sampleRate);

    const logMel = new Float32Array(nMels * nFrames);
    for (let m = 0; m < nMels; m++) {
        for (let f = 0; f < nFrames; f++) {
            let sum = 0;
            for (let k = 0; k < nFreqs; k++) {
                sum += fb[k * nMels + m] * power[k * nFrames + f];
            }
            logMel[m * nFrames + f] = 10.0 * Math.log10(Math.max(sum, 1e-10));
        }
    }

    const dct = computeDct2(logMel, nMels, nFrames);

    const mfcc = new Float32Array(nMfcc * nFrames);
    for (let m = 0; m < nMfcc; m++) {
        for (let f = 0; f < nFrames; f++) {
            mfcc[m * nFrames + f] = dct[m * nFrames + f];
        }
    }

    return { data: mfcc, rows: nMfcc, cols: nFrames };
}
