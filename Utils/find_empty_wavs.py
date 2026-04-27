import os
import wave
import struct
import math
import argparse
from pathlib import Path

def check_empty_wavs(dataset_dir, rms_threshold=10.0, peak_threshold=200):
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_path.resolve()}")
        return

    empty_files = []
    silent_files = []
    corrupt_files = []

    print(f"Scanning directory: {dataset_path.resolve()} for .wav files...")
    
    wav_count = 0
    for wav_file in dataset_path.rglob("*.wav"):
        wav_count += 1
        try:
            # 1. Check if file is literally 0 bytes
            if wav_file.stat().st_size == 0:
                empty_files.append(str(wav_file))
                continue
            
            # 2. Check using wave module for valid header and 0 frames
            with wave.open(str(wav_file), 'rb') as w:
                if w.getnframes() == 0:
                    empty_files.append(str(wav_file))
                    continue
            
            # 3. Check for absolute silence or near-silence (low RMS)
            with wave.open(str(wav_file), 'rb') as w:
                frames = w.readframes(w.getnframes())
                sampwidth = w.getsampwidth()
                
                num_samples = len(frames) // sampwidth
                if num_samples == 0:
                    empty_files.append(str(wav_file))
                    continue
                
                fmt = f"<{num_samples}h" if sampwidth == 2 else f"<{num_samples}b"
                try:
                    samples = struct.unpack(fmt, frames[:num_samples*sampwidth])
                    rms = math.sqrt(sum(s**2 for s in samples) / num_samples)
                    peak = max(abs(s) for s in samples)
                    
                    if rms < rms_threshold and peak < peak_threshold:
                        silent_files.append(str(wav_file))
                except Exception:
                    pass
                
        except Exception as e:
            # Catch files that fail to read (corrupted header, not a real wav, etc.)
            corrupt_files.append((str(wav_file), str(e)))

    print(f"Scanned {wav_count} .wav files.")

    if not empty_files and not silent_files and not corrupt_files:
        print("No empty, silent, or corrupt .wav files found!")
    
    if empty_files:
        print("\n--- Empty Files (0 bytes or 0 frames) ---")
        for f in empty_files:
            print(f)
            
    if silent_files:
        print("\n--- Silent Files (Valid audio but completely silent) ---")
        for f in silent_files:
            print(f)
            
    if corrupt_files:
        print("\n--- Corrupt/Unreadable Files ---")
        for f, err in corrupt_files:
            print(f"{f} - Error: {err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find empty or silent .wav files in a dataset.")
    parser.add_argument("--dir", type=str, help="Dataset directory (defaults to ../dataset or ./dataset)")
    parser.add_argument("--rms", type=float, default=10.0, help="RMS threshold for silence (default: 10.0)")
    parser.add_argument("--peak", type=int, default=200, help="Peak threshold for silence (default: 200)")
    
    args = parser.parse_args()

    # Resolving path assuming script is run from inside Utils/ or project root
    current_dir = Path.cwd()
    
    if args.dir:
        dataset_dir = Path(args.dir)
    elif current_dir.name == "Utils":
        dataset_dir = current_dir.parent / "dataset"
    else:
        dataset_dir = current_dir / "dataset"
        
    print(f"Thresholds: RMS < {args.rms}, Peak < {args.peak}")
    check_empty_wavs(dataset_dir, rms_threshold=args.rms, peak_threshold=args.peak)
